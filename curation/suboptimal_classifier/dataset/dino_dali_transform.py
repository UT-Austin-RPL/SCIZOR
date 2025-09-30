from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from nvidia.dali.backend import TensorListGPU
from typing import Union, Tuple, Mapping, Optional
import numpy as np
from typing import OrderedDict
import tensorflow as tf
from nvidia.dali.backend import PreallocatePinnedMemory
from octo.octo.data.dataset import make_interleaved_dataset
# import Pipe
from multiprocessing import Pipe, Process
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import ray
from nvidia.dali.plugin.jax import DALIGenericIterator
from nvidia.dali.plugin import jax as dax

# PreallocatePinnedMemory(5 * 1024 * 1024 * 1024)
import jax.numpy as jnp
import jax
devices = jax.devices()
from threading import Thread, Semaphore
import psutil
cpu_count = psutil.cpu_count()

import time
from octo.octo.utils.train_utils import Timer

timer = Timer()

def get_timer():
    return timer

def count_time_decorator(func, name:str):
    def wrapper(*args, **kwargs):
        global timer
        with timer(name):
            result = func(*args, **kwargs)
        return result
    return wrapper

def set_cpu_affinity(pid:int, cpu_id:list[int]):
    process = psutil.Process(pid)
    process.cpu_affinity(cpu_id)

def dataset_process_fn(config, child_conn, skip):
    train_data = make_interleaved_dataset(**config.dataset_kwargs, train=True, curation_args=config.curation)
    train_data.skip(skip)
    dataset_statistics = train_data.dataset_statistics
    
    child_conn.send(dataset_statistics)
    
    iterator = train_data.iterator(prefetch=config.prefetch_num_batches)
    while True:
        batch_data = next(iterator)
        child_conn.send(batch_data)

class Process_Dataset():
    def __init__(self, config, skip):
        self.child_conn, self.parent_conn = Pipe()
        self.process = Process(target=dataset_process_fn, args=(config, self.child_conn, skip), daemon=True)
        # set_cpu_affinity(self.process.pid, list(range(cpu_count//12, cpu_count)))
        self.process.start()
        self.dataset_statistics = self.parent_conn.recv()
        self.example_batch = self.parent_conn.recv()
        
    def get_example_batch(self):
        return self.example_batch
    
    def get_dataset_statistics(self):
        return self.dataset_statistics
    
    def __del__(self):
        self.process.terminate()
        
    def get_batch(self):
        return self.parent_conn.recv()
    
    def __iter__(self):
        return self
    
    def __next__(self):
        return self.parent_conn.recv()

@ray.remote(num_cpus=1)
def convert_list_bytes_to_np_array(list_of_byte_array, zero_pad_encoded):
    def convert_bytes_to_np_array(byte_array, zero_pad_encoded):            
        if len(byte_array) == 0:
            result = zero_pad_encoded
        else:
            result = np.frombuffer(byte_array, dtype=np.uint8)
        return result
    
    with ThreadPoolExecutor(max_workers=64) as executor:
        results = list(executor.map(
            partial(
                convert_bytes_to_np_array,
                zero_pad_encoded=zero_pad_encoded
                ), 
            list_of_byte_array)
        )
        return results
    
class FinalDataset():
    def __init__(self, extern_dataset, process_pipeline, buffer_device=None):
        self.extern_dataset = extern_dataset
        self.batch_size = self.extern_dataset.batch_size
        self.dataset_statistics = self.extern_dataset.dataset_statistics
        self.extern_raw_batch_queue = self.extern_dataset.raw_batch_queue
        self.process_pipeline = process_pipeline
        # self.dali_iterator = DALIGenericIterator(self.process_pipeline, output_map=self.extern_dataset.image_keys+['timestep_marker'])
        self.dali_iterator = DALIGenericIterator(self.process_pipeline, output_map=['primary', 'timestep_marker'])
        self.image_keys:list[str] = extern_dataset.image_keys
        self.buffer = []
        self.time_cnt = 0
        self.cnt = 0
        self.valid_bytes = 4
        self.buffer_device = buffer_device
        
    
    def __iter__(self):
        return self
    
    @partial(
        count_time_decorator,
        name="FinalDataset.__next__"
    )
    def __next__(self):
        self.cnt += 1
        start = time.time()
        
        while len(self.extern_raw_batch_queue) == 0:
            time.sleep(0.01)
        # batch, batch_time_step_marker = self.extern_raw_batch_queue.pop(0)
        batch, batch_time_step_marker = None, None
        processed_images = next(self.dali_iterator)
        time_step_marker = jnp.concatenate(processed_images['timestep_marker'][:self.valid_bytes], axis=0)
        
        with timer("FinalDataset.__next__ processing"):
            while batch is None:
                for i, (raw_batch, raw_time_step_marker) in enumerate(self.extern_raw_batch_queue):
                    if jnp.all(raw_time_step_marker[:self.valid_bytes] == time_step_marker):
                        batch, batch_time_step_marker = self.extern_raw_batch_queue.pop(i)
                        break
                    
            assert jnp.all(time_step_marker == batch_time_step_marker[:self.valid_bytes])
            del processed_images['timestep_marker']
            
            if self.buffer_device is not None:
                processed_images = jax.device_put(processed_images, self.buffer_device)
            
            image_dict = {}
            for i,key in enumerate(self.image_keys['primary']):
                image_dict[key] = processed_images['primary'][i*self.batch_size : (i+1) * self.batch_size]
                
            # for i,key in enumerate(self.image_keys['wrist']):
            #     image_dict[key] = processed_images['wrist'][i*self.batch_size : (i+1) * self.batch_size]
            
            for key in image_dict:
                splited_keys = key.split('.')
                current_data = batch
                for j, splited_key in enumerate(splited_keys[:-1]):
                    # if splited_key is not a number
                    if splited_key[:6] == "image_":
                        if splited_keys[0] == "task":
                            current_data[splited_key] = image_dict[key]
                        else: # observation
                            if int(splited_keys[j+1]) == 0:
                                current_data[splited_key] = image_dict[key][:, jnp.newaxis]
                            else:
                                current_data[splited_key]= jnp.concatenate([current_data[splited_key], image_dict[key][:, jnp.newaxis]], axis=1)
                            
                    else:
                        current_data = current_data[splited_key]
                        
        return batch
                
                

class ExternIterator(object):
    def __init__(self, config, max_size=10, skip=0):
        self.dataset = Process_Dataset(config, skip=skip)
        self.dataset_statistics = self.dataset.get_dataset_statistics()
        self.iterator = self.dataset.__iter__()
        self.batch_size = config.dataset_kwargs.batch_size
        self.example_batch = next(self.iterator)
        # self.task_image_keys = [key for key in self.example_batch['task'] if key.startswith("image_")]
        self.observation_image_keys = ['image_primary']
        self.zero_pad = tf.zeros((256, 256, 3), dtype=tf.uint8)
        self.zero_pad_encoded = tf.image.encode_jpeg(self.zero_pad).numpy()
        self.zero_pad_encoded = np.frombuffer(self.zero_pad_encoded, dtype=np.uint8)
        self.window_size = config.window_size
        self.time_cnt = 0
        self.cnt = 0
        self.image_keys = None
        self.raw_batch_queue = []
        self.queue = []
        self.max_size = max_size
        self.get_image_keys()
        
        ray.init(log_to_driver=False)
        self.queue_semaphore = Semaphore(0)
        self.thread = Thread(target=self.fetch_next, daemon=True)
        self.thread.start()
        
        
    def __iter__(self):
        return self
    
    def get_num_outputs(self):
        return len(self.observation_image_keys) * self.window_size
    
    def fetch_next(self):
        while True:
            result = self.next()
            self.queue.append(result)
            self.queue_semaphore.release()
    
    @partial(
        count_time_decorator,
        name="ExternIterator.__next__"
    )
    def __next__(self):
        self.queue_semaphore.acquire()
        result = self.queue.pop(0)
        return result
    
    def get_image_keys(self):
        self.image_keys = {'primary':[]}
        for key in self.observation_image_keys:
            for i in range(self.window_size):
                self.image_keys[key[6:]].append(f"observation.{key}.{i}")
    
    def next(self):
        # if self.cnt % 10 == 0:
        #     print(len(self.queue))
        def create_time_step_marker():
            marker = np.random.randint(0, 255, self.batch_size, dtype=np.uint8)
            marker[0] = self.cnt % 256
            return marker
        
        while len(self.raw_batch_queue) > self.max_size:
            time.sleep(0.02)
        self.cnt += 1
        
        batch_to_transform = OrderedDict()
        futures = OrderedDict()
        
        for key in self.observation_image_keys:
            for i in range(self.window_size):
                batch_to_transform[f"observation.{key}.{i}"] = []
                
        # if self.image_keys is None:
        #     self.image_keys = list(batch_to_transform.keys())
                
        frame = next(self.iterator)
        del frame['observation']['image_wrist']
        time_step_marker = create_time_step_marker()
        self.raw_batch_queue.append((frame, time_step_marker))
        
        for key in self.observation_image_keys:
            for i in range(self.window_size):
                futures[f"observation.{key}.{i}"] = convert_list_bytes_to_np_array.remote(frame['observation'][key][:,i].tolist(), self.zero_pad_encoded)
        
        flattened_images = []
        flattened_mask = {'primary':[]}
        images = {'primary':[]}
            
        for key in self.observation_image_keys:
            for i in range(self.window_size):
                batch_to_transform[f"observation.{key}.{i}"] = ray.get(futures[f"observation.{key}.{i}"])
                images[key[6:]].extend(batch_to_transform[f"observation.{key}.{i}"])
            
            
        for key in self.observation_image_keys:
            for i in range(self.window_size):
                batch_to_transform[f"observation.pad_mask_dict.{key}.{i}"] = np.split(np.array(frame['observation']['pad_mask_dict'][key][:,i], dtype=np.uint8), self.batch_size, axis=0)
                flattened_mask[key[6:]].extend(batch_to_transform[f"observation.pad_mask_dict.{key}.{i}"])
        
        batch_to_transform['timestep'] = np.split(time_step_marker, self.batch_size, axis=0) * len(self.image_keys['primary'])

        return images['primary'], flattened_mask['primary'], batch_to_transform['timestep']
    
    def __del__(self):
        ray.shutdown()

def augment_brightness(
    input, scale: list[float]
):
    rand_shift = fn.random.uniform(range=(-scale[0], scale[0]))
    result = fn.brightness(input, brightness_shift=rand_shift)
    return result

def augment_contrast(
    input, scale: tuple[float, float]
):
    rand_scale = fn.random.uniform(range=[scale[0], scale[1]])
    result = fn.contrast(input, contrast=rand_scale)
    return result

def augment_brightness_contrast(
    input, bright_scale: tuple[float], contrast_scale: tuple[float, float]
):
    rand_shift = fn.random.uniform(range=(-bright_scale[0], bright_scale[0]))
    rand_scale = fn.random.uniform(range=[contrast_scale[0], contrast_scale[1]])
    
    result = fn.brightness_contrast(input, brightness_shift=rand_shift, contrast=rand_scale)
    return result

def augment_saturation(
    input, scale: tuple[float, float]
):
    rand_scale = fn.random.uniform(range=[scale[0], scale[1]])
    result = fn.saturation(input, saturation=rand_scale)
    return result

def augment_hue(
    input, scale: list[float]
):
    rand_shift = fn.random.uniform(range=[-scale[0], scale[0]])
    result = fn.hue(input, hue=rand_shift)
    return result

def augment_random_resized_crop(
    input, ratio, scale, size
):
    result = fn.random_resized_crop(input, size=size, random_aspect_ratio=ratio, random_area=scale)
    return result

@pipeline_def
def test_pipeline(image_augment_kwargs):
    files, label = fn.readers.file(file_root="")
    images = fn.decoders.image(files, device="cpu")
    # converted = augment_contrast(images, scale=[0.5, 0.6])
    augment_arg = image_augment_kwargs.get('primary', {'augment_order': []})
    for augment_op in augment_arg['augment_order']:
        if augment_op in AUGMENT_OPS and augment_arg.get(augment_op) is not None:
            if augment_op == "random_resized_crop":
                images = AUGMENT_OPS[augment_op](images, **augment_arg[augment_op], size=(256,256))
            else:
                images = AUGMENT_OPS[augment_op](images, augment_arg[augment_op])
    # print("Converted: ", converted)
    
    return images

AUGMENT_OPS = {
    # "random_resized_crop": augment_random_resized_crop,
    # "random_brightness": augment_brightness,
    # "random_contrast": augment_contrast,
    # "random_brightness_contrast": augment_brightness_contrast,
    # "random_saturation": augment_saturation,
    # "random_hue": augment_hue,
}

@pipeline_def(enable_conditionals=True)
def dali_obs_transform_pipeline(
    source: ExternIterator,
    device: str,
    image_augment_kwargs: Union[dict, Mapping[str, dict]] = {},
    resize_size: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]] = {},
    depth_resize_size: Union[Tuple[int, int], Mapping[str, Tuple[int, int]]] = {},
    image_dropout_prob: float = 0.0,
    image_dropout_keep_key: Optional[str] = None,
    num_parallel_calls: int = 1,
    save_gpu_memory: bool = False,
):    
    assert device in ['cpu', 'mixed']
    num_images = 1
    frame = fn.external_source(
        source=source, num_outputs=3, device='cpu', dtype=types.UINT8, 
    )
    key_list = ["primary"]
    for i,current_key in enumerate(key_list):
        # current_key = key_list[i]
        current_pad_mask = frame[i+num_images]
        
        #Ignore the depth since load_depth is False in the config
        
        #Decode the image
        frame[i] = fn.decoders.image(frame[i], device='cpu' if save_gpu_memory else device)
        
        #Resize the image
        if current_key in resize_size:
            # print("Resizing to ", resize_size[current_key])
            frame[i] = fn.resize(
                frame[i],
                resize_x=resize_size[current_key][0],
                resize_y=resize_size[current_key][1],
                device='cpu' if device == 'cpu' or save_gpu_memory else 'gpu',
            )  
            
        # Ignore the dropout since dropout prob is 0 in the config
        
        
        augment_arg = image_augment_kwargs.get(current_key, {'augment_order': []})
        
        frame[i] = frame[i].gpu() if device == 'mixed' else frame[i]
        current_pad_mask = current_pad_mask.gpu() if device == 'mixed' else current_pad_mask

        for augment_op in augment_arg['augment_order']:
            if augment_op in AUGMENT_OPS and augment_arg.get(augment_op) is not None:
                if augment_op == "random_resized_crop":
                    frame[i] = AUGMENT_OPS["random_resized_crop"](frame[i], **augment_arg["random_resized_crop"], size=resize_size[current_key])
                else:
                    frame[i] = AUGMENT_OPS[augment_op](frame[i], augment_arg[augment_op])
           
        frame[i] *= current_pad_mask
        
    # the last element is timestep marker
    return tuple(frame[:num_images]) + (frame[-1],)
    
class InterleaveDaliDataset:
    def __init__(
        self,
        config,
        source: ExternIterator,
        num_threads_per_dataset: int = 64,
        device: str = 'mixed' if len(devices) > 1 else 'cpu',
        num_device: int = len(devices),
        num_dataset: int = len(devices) if len(devices) > 1 else 1,
        prefetch_queue_depth: int = 2,
        save_gpu_memory: bool = False,
        transform_fn = lambda x: x,
        buffer_size: int = 2,
        buffer_device: str = None, # the last device
        device_ids: list[int] = None,
    ):
        assert device in ['cpu', 'mixed']
        self.config = config
        self.obs_transform_pipeline = []
        self.train_datasets = []
        self.train_data_iters = []
        self.num_device = num_device
        self.num_dataset = num_dataset
        self.source = source
        self.prefetch_queue_depth = prefetch_queue_depth
        self.step_cnt = 0
        self.num_threads_per_dataset = num_threads_per_dataset
        self.device_ids = []
        self.buffer = []
        self.device = device
        
        if buffer_device is None:
            self.buffer_device = None
        else:
            if buffer_device == 'cpu':
                self.buffer_device = jax.devices('cpu')[0]
            elif buffer_device.isdigit():
                self.buffer_device = jax.devices()[buffer_device]
            else:
                raise ValueError(f"Invalid Buffer device: {buffer_device}, buffer device should be either 'cpu' or a number")
        
        for i in range(num_dataset):
            if device_ids is not None and len(device_ids) == num_device:
                device_id = device_ids[i%num_device]
            else:
                device_id = i % num_device if device == 'mixed' else None
                
            self.device_ids.append(device_id)
            self.obs_transform_pipeline.append(dali_obs_transform_pipeline(source=source,
                                                            batch_size=config.dataset_kwargs.batch_size * len(source.image_keys['primary']), 
                                                            num_threads=num_threads_per_dataset,
                                                            device=device, 
                                                            device_id=device_id,
                                                            **config.dataset_kwargs["frame_transform_kwargs"],
                                                            prefetch_queue_depth=prefetch_queue_depth,
                                                            save_gpu_memory=save_gpu_memory,
                                                            ))
            self.obs_transform_pipeline[-1].build()
            self.train_datasets.append(FinalDataset(source, self.obs_transform_pipeline[-1], buffer_device=self.buffer_device))
            self.train_data_iters.append(self.train_datasets[-1].__iter__())
            self.example_batch = next(self.train_datasets[-1])
            
        def fill_buffer_thread():
            while True:
                if len(self.buffer) <= buffer_size:
                    batch = next(self.train_data_iters[self.step_cnt % self.num_dataset])
                    batch = transform_fn(batch)
                    self.buffer.append(batch)
                    self.step_cnt += 1
                else:
                    time.sleep(0.05)
                
        self.fill_buffer_thread = Thread(target=fill_buffer_thread, daemon=True)
        self.fill_buffer_thread.start()
                    
        
            
    def __iter__(self):
        return self
    
    def __next__(self):
        while len(self.buffer) == 0:
            time.sleep(0.01)
        result = self.buffer.pop(0)
        return result