from curation.suboptimal_classifier.dataset.dino_dali_transform import InterleaveDaliDataset, ExternIterator, count_time_decorator, get_timer
from curation.suboptimal_classifier.dataset.rlds.oxe import make_oxe_dataset_kwargs_and_weights
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from curation.suboptimal_classifier.dataset.augment_action import augment_action

from absl import flags, app
from ml_collections import config_flags
import os
import os.path as osp
import tensorflow as tf
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from curation.suboptimal_classifier.utils import process_char_array_for_dino, normalize_images
from curation.suboptimal_classifier.dataset.rlds.dataset import make_interleaved_dataset
from transformers import AutoImageProcessor, AutoModel, CLIPImageProcessor
import zmq
import pickle
import h5py
from curation.suboptimal_classifier.dataset.hdf5_dataset import HDF5Dataset

FLAGS = flags.FLAGS

#disable tf gpu
tf.config.set_visible_devices([], 'GPU')

timer = get_timer()

class AugActionDataset():
    def __init__(self, config, *args, **kwargs):
        self.oxe_dataset_kwargs_list = config.dataset_kwargs.get("dataset_kwargs_list", None)
        if self.oxe_dataset_kwargs_list is not None:
            self.dataset = make_interleaved_dataset(**config.dataset_kwargs, curation_args=config.curation, train=True, use_dali=False)
            self.data_iter = self.dataset.iterator(prefetch=1)
        self.action_aug_kwargs = config.discriminator_dataset_kwargs.action_aug_kwargs
        self.obs_aug_kwargs = config.discriminator_dataset_kwargs.obs_aug_kwargs
        self.image_key = config.discriminator_dataset_kwargs.image_key
        self.proprio_noise_scale = config.discriminator_dataset_kwargs.proprio_noise_scale
        self.action_noise_scale = config.discriminator_dataset_kwargs.action_noise_scale
        self.history_aug_prob = config.discriminator_dataset_kwargs.history_aug_prob
        self.action_horizon = config.action_horizon
        self.future_action = config.future_action
        self.window_size = config.window_size
        self.subop_dataset_path = config.subop_dataset.path
        self.future_image = config.future_image
        self.action_aug = config.discriminator_dataset_kwargs.action_aug
        self.no_action_input = config.discriminator.no_action_input
        self.head_type = config.discriminator.head_type
        self.goal_relabeling_strategy = config.dataset_kwargs.traj_transform_kwargs.goal_relabeling_strategy
        if self.goal_relabeling_strategy == "custom_uniform":
            self.future_image_stat = {'max': config.dataset_kwargs.traj_transform_kwargs.goal_relabeling_kwargs.max_goal_time_diff, 'min': config.dataset_kwargs.traj_transform_kwargs.goal_relabeling_kwargs.min_goal_time_diff}
        else:
            self.future_image_stat = None
        self.encoder_type = config.discriminator.encoder_type
        if self.encoder_type == "dinov2":
            self.processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
        elif self.encoder_type == "radio":
            self.processor =  CLIPImageProcessor.from_pretrained('nvidia/RADIO-B')
        self.hdf5_interleave_dataset_loader = None
        
        if self.subop_dataset_path is not None:
            self.hdf5_dataset = Hdf5SubopDataset(self.subop_dataset_path, window_size=self.window_size, action_horizon=self.action_horizon, future_action=self.future_action, future_image=self.future_image)
            self.hdf5_interleave_dataset = Hdf5SubopInterleaveDataset(self.hdf5_dataset, metrics=config.subop_dataset.metrics, dataset_names=config.subop_dataset.dataset_names)
            if len(self.hdf5_interleave_dataset) == 0:
                print("-"*40)
                print("Warning: No data in subop dataset, skipping")
            else:
                if config.subop_dataset.label_balance:
                    labels = self.hdf5_interleave_dataset.get_all_labels()
                    
                    weights = np.zeros_like(labels, dtype=np.float32)
                    
                    subdataset_size = self.hdf5_interleave_dataset.get_subdataset_size()
                    for i in range(len(subdataset_size)):
                        start, end = sum(subdataset_size[:i]), sum(subdataset_size[:i+1])
                        current_label = labels[start:end]
                        num_subop = np.sum(current_label) + 1e-6
                        num_good = len(current_label) - num_subop + 1e-6
                        num_subop = np.clip(num_subop, num_good/10, num_good*10)
                        weights[start:end] = np.where(current_label==1, len(current_label)/num_subop, len(current_label)/num_good)
                    
                    sampler = WeightedRandomSampler(weights, len(weights))
                else:
                    sampler = None
                self.hdf5_interleave_dataset_loader = DataLoader(self.hdf5_interleave_dataset, batch_size=config.subop_dataset.batch_size, shuffle=sampler is None, num_workers=8, drop_last=True, sampler=sampler)
                self.hdf5_interleave_dataset_iterator = iter(self.hdf5_interleave_dataset_loader)
                print("-"*40)
                print(f"Loaded subop dataset from {self.subop_dataset_path}")
                
        print(f"Action Horizon: {self.action_horizon}, Window Size: {self.window_size}, Future Action: {self.future_action}, Future Image: {self.future_image}")
        
    def __next__(self):
        """
        Returns a batch of images, actions, action_scores, and task descriptions
        images: B, T, C, H, W
        actions: B, T, 7
        action_scores: B, T
        task_desc: B
        """
        if self.oxe_dataset_kwargs_list is not None:
            result = next(self.data_iter)
            observations = result['observation']
            observations[self.image_key] = torch.tensor(observations[self.image_key])
                
            # remove action chunking, at most one action per observation, and only pick action_history number of actions
            actions = result['action'][:, -self.action_horizon:, 0].copy() # B, H, 7
            future_action = result['action'][:, -1, 1:self.future_action+1].copy() # B, future_action, 7
            B, H, _ = actions.shape
            norm_delta_proprio = result['observation']['delta_proprio'][:, -self.action_horizon:].copy() # B, H
            # future_norm_delta_proprio = result['observation']['future_delta_proprio'][:, -1, 1:self.future_action+1] # B, future_action
            delta_grip_act = result['observation']['delta_gripping'][:, -self.action_horizon:].copy() # B, H
            
            # current_device = observations[image_key].device
            current_device = torch.device('cpu')
            
            task_desc = result['task']['language_instruction'].copy()
            
            if len(task_desc.shape) > 1:
                task_desc = task_desc[:,0]
            
            if self.obs_aug_kwargs['dropout_image_prob'] > 0:
                # for key in observations:
                #     if 'image' in key:
                    dropout = torch.rand(observations[self.image_key].shape[:2], device=current_device) < self.obs_aug_kwargs['dropout_image_prob']
                    observations[self.image_key][dropout] = 0
                        
            if self.obs_aug_kwargs['text_add_front_prob'] > 0:
                add_gripper = np.random.random(task_desc.shape) < self.obs_aug_kwargs['text_add_front_prob']
                task_desc[add_gripper] = self.obs_aug_kwargs['text_front'].encode() + task_desc[add_gripper]
                is_empty = task_desc == b""
                task_desc[is_empty] = self.obs_aug_kwargs['text_front'].encode() + task_desc[is_empty] # add gripper to empty task descriptions to prevent errors
            
            task_desc = process_char_array_for_dino(task_desc)
            
            images = observations[self.image_key]
            if self.future_image:
                future_image = torch.tensor(result['task'][self.image_key])
                future_image_shape = future_image.shape
                num_future_images = future_image_shape[1]
                images = images[:, None].repeat(1, num_future_images, 1, 1, 1, 1)
                future_image = future_image[:, :, None]
                images = torch.cat([images, future_image], dim=2)
                delta_time = result['task']['delta_time']
                if self.future_image_stat is not None:
                    delta_time = (delta_time - self.future_image_stat['min'])/(self.future_image_stat['max'] - self.future_image_stat['min'])
                    
                if self.head_type == "regression":
                    image_score = 1 - delta_time
                elif self.head_type == "rank":
                    image_score = delta_time
            elif self.goal_relabeling_strategy == "uniform_action":
                action_score = result['task']['delta_time']!=0
                
    
            image_shape = images.shape
            images = images.view(-1, *image_shape[-3:]) # B*T, H, W, C
            if self.encoder_type == "dinov2" or self.encoder_type == "radio":
                images = self.processor(images, return_tensors='pt')['pixel_values']
                images = images.view(*image_shape[:-3], *images.shape[1:])
            else:
                images = normalize_images(images)
                images = images.view(*image_shape[:-3], *images.shape[1:])
                images = images.transpose(-1, -3)
                # if len(images.shape) == 5:
                #     images = images.permute(0, 1, 4, 2, 3) # B, T, C, H, W
                #     images = images[:, -(self.window_size+self.future_image):] # B, T, C, H, W               
                    
                # elif len(images.shape) == 6:
                #     images = images.permute(0, 1, 2, 5, 3, 4) # B, num_future_images, T, C, H, W
                #     images = images[:, :, -(self.window_size+self.future_image):] # B, num_future_images, T, C, H, W      
            if len(images.shape) == 5:
                # images = images.permute(0, 1, 4, 2, 3) # B, T, C, H, W
                images = images[:, -(self.window_size+self.future_image):] # B, T, C, H, W               
                
            elif len(images.shape) == 6:
                # images = images.permute(0, 1, 2, 5, 3, 4) # B, num_future_images, T, C, H, W
                images = images[:, :, -(self.window_size+self.future_image):] # B, num_future_images, T, C, H, W          
                
            
            aug_actions, aug_action_scores, sub_scores, aug_norm_delta_proprio, aug_delta_grip_act = augment_action(actions, norm_delta_proprio, delta_grip_act, self.action_aug_kwargs.to_dict()) # only augment the last action
            history_aug = np.random.rand(B, H-1) < self.history_aug_prob
            to_aug = np.concatenate([history_aug, np.ones((B, 1), dtype=bool)], axis=1)
            aug_action_scores, sub_scores, aug_actions, aug_norm_delta_proprio, aug_delta_grip_act = aug_action_scores.numpy(), sub_scores.numpy(), aug_actions.numpy(), aug_norm_delta_proprio.numpy(), aug_delta_grip_act.numpy()
            
            if not self.action_aug or self.no_action_input:
                to_aug = np.zeros_like(to_aug)
                aug_action_scores = np.zeros_like(aug_action_scores)
                sub_scores = np.zeros_like(sub_scores)
            
            actions[to_aug], norm_delta_proprio[to_aug], delta_grip_act[to_aug] = aug_actions[to_aug], aug_norm_delta_proprio[to_aug], aug_delta_grip_act[to_aug]
            
            scores = aug_action_scores
            
            if self.future_image:
                scores = scores[:, None].repeat(repeats=num_future_images, axis=1)
                scores += image_score
            if self.goal_relabeling_strategy == "uniform_action":
                scores += action_score
            
        else:
            actions = np.zeros((0, self.action_horizon, 7), dtype=np.float32)
            future_action = np.zeros((0, self.future_action, 7), dtype=np.float32)
            norm_delta_proprio = np.zeros((0, self.action_horizon), dtype=np.float32)   
            delta_grip_act = np.zeros((0, self.action_horizon), dtype=np.float32)
            images = torch.zeros((0, self.window_size+self.future_image, 3, 256, 256), dtype=torch.float32)
            task_desc = []
            scores = np.zeros((0), dtype=np.float32)
            sub_scores = np.zeros((0, 4), dtype=np.float32)
            current_device = torch.device('cpu')
            
            assert self.hdf5_interleave_dataset_loader is not None, "No dataset path provided"
            
        if self.hdf5_interleave_dataset_loader is not None:
            try:
                subop_batch = next(self.hdf5_interleave_dataset_iterator)
            except StopIteration:
                self.hdf5_interleave_dataset_iterator = iter(self.hdf5_interleave_dataset_loader)
                subop_batch = next(self.hdf5_interleave_dataset_iterator)
            for item in subop_batch:
                if isinstance(item, torch.Tensor):
                    item = item.numpy()
            subop_images, subop_actions, subop_task_desc, subop_norm_delta_proprio, subop_delta_grip_act, label = subop_batch
            # breakpoint()
            # subop_scores = np.ones((subop_actions.shape[0]), dtype=np.float32)
            subop_scores = np.array(label, dtype=np.float32)
            subop_sub_scores = np.zeros((subop_actions.shape[0], 4), dtype=np.float32) # hardcode for now
            subop_actions, subop_future_action = subop_actions[:, :self.action_horizon], subop_actions[:, self.action_horizon:]
            
            images = torch.cat([images, subop_images], dim=0)
            actions = np.concatenate([actions, subop_actions], axis=0)
            future_action = np.concatenate([future_action, subop_future_action], axis=0)
            task_desc = task_desc + list(subop_task_desc)
            # discard future pad for subop dataset
            norm_delta_proprio = np.concatenate([norm_delta_proprio, subop_norm_delta_proprio[:, :self.action_horizon]], axis=0)
            delta_grip_act = np.concatenate([delta_grip_act, subop_delta_grip_act[:, :self.action_horizon]], axis=0)
            B, H, _ = actions.shape
    
            scores = np.concatenate([scores, subop_scores], axis=0)
            sub_scores = np.concatenate([sub_scores, subop_sub_scores], axis=0)
            
        actions =  np.concatenate([actions, future_action], axis=1).astype(np.float32)
        
        actions += np.random.normal(0, self.action_noise_scale, actions.shape)
        norm_delta_proprio += np.random.normal(0, self.proprio_noise_scale, norm_delta_proprio.shape)
        
        future_pad = np.zeros((B, self.future_action))
        norm_delta_proprio = np.concatenate([norm_delta_proprio, future_pad], axis=1).astype(np.float32)
        delta_grip_act = np.concatenate([delta_grip_act, future_pad], axis=1).astype(np.float32)    
        
        actions, scores, sub_scores, norm_delta_proprio, delta_grip_act = torch.tensor(actions).to(current_device), torch.tensor(scores).to(current_device), torch.tensor(sub_scores).to(current_device), torch.tensor(norm_delta_proprio).to(current_device), torch.tensor(delta_grip_act).to(current_device)         
        
        return images, actions, scores, sub_scores, task_desc, norm_delta_proprio, delta_grip_act
    
    def next(self):
        return self.__next__()
    
    def skip(self, steps):
        for _ in range(steps):
            while len(self.buffer) == 0:
                time.sleep(0.01)
            self.buffer.pop(0)

class Hdf5SubopSubDataset(Dataset):
    def __init__(self, hdf5, window_size=1, action_horizon=1, future_action=0, future_image=False, zero_score_only=False, one_score_only=False):
        self.hdf5 = hdf5
        self.window_size = window_size
        self.action_horizon = action_horizon
        self.future_action = future_action
        self.future_image = future_image
        self.zero_score_only = zero_score_only
        self.one_score_only = one_score_only
        if zero_score_only:
            self.idx_mask = self.hdf5['label'][:] == 0.0
        elif one_score_only:
            self.idx_mask = self.hdf5['label'][:] == 1.0
        else:
            self.idx_mask = np.ones(len(self.hdf5['image']), dtype=bool)
        self.new_idx = np.argwhere(self.idx_mask).flatten()
        
    def __len__(self):
        return np.sum(self.idx_mask)
    
    def __getitem__(self, idx):
        idx = self.new_idx[idx]
        image = self.hdf5['image'][idx]
        if self.future_image:
            future_image = self.hdf5['future_image'][idx][None]
            image = np.concatenate([image, future_image], axis=0)
        image = torch.tensor(image)
        if len(image.shape) == 3:
            image = image.unsqueeze(0)
        # bgr_to_rgb_idx = [2, 1, 0]
        # image = image[..., bgr_to_rgb_idx] # invert image channels, BGR -> RGB
        image = normalize_images(image, batched=True)
        image = image.permute(0, 3, 1, 2) # window_size, C, H, W
        image = image[-(self.window_size+self.future_image):]
        action = self.hdf5['action'][idx]
        future_action = self.hdf5['future_action'][idx]
        norm_delta_proprio = self.hdf5['norm_delta_proprio'][idx]
        norm_delta_proprio = torch.tensor(norm_delta_proprio)
        delta_grip_act = self.hdf5['delta_grip_act'][idx]
        delta_grip_act = torch.tensor(delta_grip_act)
        if len(action.shape) == 1:
            action = action[None] # H, 7
        action = action[-self.action_horizon:]
        future_action = future_action[:self.future_action]
        action = torch.tensor(np.concatenate([action, future_action], axis=0))
        
        future_pad = torch.zeros((self.future_action))
        norm_delta_proprio = norm_delta_proprio[-self.action_horizon:]
        norm_delta_proprio = torch.cat([norm_delta_proprio, future_pad], dim=0)
        delta_grip_act = delta_grip_act[-self.action_horizon:]
        delta_grip_act = torch.cat([delta_grip_act, future_pad], dim=0)
        
        language_instruction = self.hdf5['language_instruction'][idx:idx+1]
        language_instruction = np.char.add(b"robot arm . gripper . ", language_instruction)
        language_instruction = process_char_array_for_dino(language_instruction)[0]
        
        if 'label' in self.hdf5:
            label = self.hdf5['label'][idx] if 'label' in self.hdf5 else 1
            label = torch.tensor(label)
        else:
            label = torch.tensor(0)
        return image, action, language_instruction, norm_delta_proprio, delta_grip_act, label
    
    def get_labels(self, idx):
        assert  'label' in self.hdf5, "No label in hdf5"
        idx = self.new_idx[idx]
        return self.hdf5['label'][idx]

class Hdf5SubopDataset():
    def __init__(self, path, window_size=1, action_horizon=1, future_action=0, future_image=False, zero_score_only=False, one_score_only=False):
        self.hdf5_file = h5py.File(path, 'r')
        self.sub_datasets = {}
        for key in self.hdf5_file:
            self.sub_datasets[key] = {}
            self.metrics = self.hdf5_file[key].keys()
            for metric in self.hdf5_file[key]:
                self.sub_datasets[key][metric] = Hdf5SubopSubDataset(self.hdf5_file[key][metric], window_size, action_horizon, future_action, future_image, zero_score_only, one_score_only)
            
        
    def num_datasets(self):
        return len(self.hdf5_file)
    
    def keys(self):
        return self.hdf5_file.keys()
    
    def all_dataset(self):
        for dataset_name in self.sub_datasets:
            for metric in self.sub_datasets[dataset_name].keys():
                if len(self.sub_datasets[dataset_name][metric]) == 0:
                    continue
                yield dataset_name, metric, self.sub_datasets[dataset_name][metric]

class Hdf5SubopInterleaveDataset():
    def __init__(self, hdf5_dataset, batch_size=32, metrics:str=None, dataset_names:str=None):
        self.hdf5_dataset = hdf5_dataset
        self.batch_size = batch_size
        self.dataset_index = []
        self.total_len = 0
        metrics = metrics.split(",") if metrics is not None else None
        dataset_names = dataset_names.split(",") if dataset_names is not None else None
        for dataset_name, metric, dataset in self.hdf5_dataset.all_dataset():
            if (dataset_names is None or dataset_name in dataset_names) and (metrics is None or metric in metrics):
                self.dataset_index.append((dataset_name, metric, len(dataset), dataset))
                self.total_len += len(dataset)
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        for dataset_name, metric, length, dataset in self.dataset_index:
            if idx < length:
                return dataset[idx]
            else:
                idx -= length
        raise IndexError("Index out of range")
    
    def get_labels(self, idx):
        for dataset_name, metric, length, dataset in self.dataset_index:
            if idx < length:
                return dataset.get_labels(idx)
            else:
                idx -= length
        raise IndexError("Index out of range")
    
    def get_all_labels(self):
        labels = []
        for dataset_name, metric, length, dataset in self.dataset_index:
            labels.append(dataset.get_labels(range(length)))
        labels = np.concatenate(labels, axis=0)
        return labels
    
    def get_subdataset_size(self):
        sub_length = []
        for dataset_name, metric, length, dataset in self.dataset_index:
            sub_length.append(length)
        return sub_length
        
        
class ZmqDataset():
    def __init__(self, config, port, *args, **kwargs):
        self.window_size = config.window_size
        if config.action_horizon > config.window_size: # make sure enough history is available for action horizon
            config.window_size = config.action_horizon
        
        if config.dataset_kwargs.oxe_kwargs.data_dir is not None:
            # create dataset_kwargs_list from oxe_kwargs
            (
                config.dataset_kwargs["dataset_kwargs_list"],
                config.dataset_kwargs["sample_weights"],
            ) = make_oxe_dataset_kwargs_and_weights(
                **config.dataset_kwargs["oxe_kwargs"]
            )
            del config.dataset_kwargs["oxe_kwargs"]
            
            # config.window_size = self.window_size
            tf.random.set_seed(config.seed + int(port))
            self.dataset = AugActionDataset(config, *args, **config.dali_kwargs)
        elif config.hdf5_dataset_kwargs.data_dir is not None:
            self.dataset = HDF5Dataset(config)
            self.image_key = config.discriminator_dataset_kwargs.image_key
        # self.action_aug_kwargs = config.discriminator_dataset_kwargs.action_aug_kwargs
        # self.obs_aug_kwargs = config.discriminator_dataset_kwargs.obs_aug_kwargs
        # self.image_key = config.discriminator_dataset_kwargs.image_key
        self.num_steps = config.num_steps
        self.num_devices = torch.cuda.device_count()

        self.torch_buffer_device = torch.device(f'cpu')
        
        self.context = zmq.Context()
        #producer for many consumers
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.bind(f"tcp://localhost:{port}")
        
        
    def run(self):
        for _ in range(self.num_steps*self.num_devices):
            # while len(self.zmq_buffer) == 0:
            #     time.sleep(0.01)
            batch = next(self.dataset)
            client_id, message = self.socket.recv_multipart()
            pickle_batch = pickle.dumps(batch, protocol=pickle.HIGHEST_PROTOCOL)
            self.socket.send_multipart([client_id, b'', pickle_batch])
            

def run_zmq_dataset(config, port):
    zmq_dataset = ZmqDataset(config, port)
    zmq_dataset.run()

def main(_):
    from tqdm import tqdm
    config = FLAGS.config
    if config.dataset_kwargs.oxe_kwargs.data_dir is not None:
        if "oxe_kwargs" in FLAGS.config.dataset_kwargs:
            # create dataset_kwargs_list from oxe_kwargs
            (
                FLAGS.config.dataset_kwargs["dataset_kwargs_list"],
                FLAGS.config.dataset_kwargs["sample_weights"],
            ) = make_oxe_dataset_kwargs_and_weights(
                **FLAGS.config.dataset_kwargs["oxe_kwargs"]
            )
            del FLAGS.config.dataset_kwargs["oxe_kwargs"]
        
    from curation.suboptimal_classifier.utils import Torch_Preallocater
    
    # preallocator = Torch_Preallocater(0.75)
    window_size = config.window_size
    if config.action_horizon > config.window_size: # make sure enough history is available for action horizon
        config.window_size = config.action_horizon
    # extern_source = ExternIterator(config) if "dataset_kwargs_list" in config.dataset_kwargs else None
    config.window_size = window_size
    # aug_action_dataset = AugActionDataset(config, **config.dali_kwargs)
    aug_action_dataset = HDF5Dataset(config)
    
    images, actions, action_scores, sub_scores, task_desc, norm_delta_proprio, delta_grip_act = next(aug_action_dataset)
    # preallocator.free()
    scores = []
    start = time.time()
    number_of_bins = 10
    eps = 1e-6
    bins = np.linspace(0.0, 1, number_of_bins+1)
    num_in_bins = np.zeros(number_of_bins+1)
    for i in tqdm(range(100)):
        images, actions, action_scores, sub_scores, task_desc, norm_delta_proprio, delta_grip_act = next(aug_action_dataset)
        # if torch.isnan(norm_delta_proprio).any():
        #     print("Nan in action scores")
        # breakpoint()
        scores.append(action_scores.cpu().numpy())  
    
    scores = np.stack(scores)
    bined_scores = np.digitize(scores, bins, right=True)
    num_in_bins += np.bincount(bined_scores.flatten(), minlength=number_of_bins+1)[:number_of_bins+1]
    print(f"Average Time taken: {(time.time()-start)/100}") 
    print(f"Average Score: {np.mean(scores)}, Max Score: {np.max(scores)}, Min Score: {np.min(scores)}, Std Score: {np.std(scores)}")
    for i in range(number_of_bins+1):
        if i==0:
            print(f"Number of scores i==0: {num_in_bins[i]}, Percentage: {num_in_bins[i]/np.sum(num_in_bins):.2f}")
        else:
            print(f"Number of scores in bin {bins[i-1]:.2f}:{bins[i]:.2f}: {num_in_bins[i]}, Percentage: {num_in_bins[i]/np.sum(num_in_bins):.2f}")
    
if __name__ == '__main__':
    flags.DEFINE_string("name", "experiment", "Experiment name.")
    flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")

    config_dir = os.path.join(os.path.dirname(__file__), "../config")
    config_flags.DEFINE_config_file(
        "config",
        os.path.join(config_dir, "config.py"),
        "File path to the training hyperparameter configuration.",
        lock_config=False,
    )
    app.run(main)
    