import torch
from cosmos1.models.tokenizer.inference.video_lib import CausalVideoTokenizer
import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import dlimp as dl
from curation.utils.oxe_dataset_configs import OXE_DATASET_CONFIGS, OXE_DATASET_CONTROL_FREQUENCY
from curation.utils.oxe_standardization_transforms import OXE_STANDARDIZATION_TRANSFORMS
from tqdm import tqdm
import tyro
import os
import pickle
import ray
import time
from threading import Thread
from utils import get_statistics
import json


image_keys = ['image']
store_keys = ['id', 'action', 'proprio', 'normalized_action']

def quantile_normalize(data, axis=0, q_low=1, q_high=99, epsilon=1e-8):
    """
    Normalize data such that the 1st and 99th quantile map to [-1, 1].
    
    Parameters:
        data (numpy array): The input data of shape (N, 8, 7).
        axis (int): The axis along which to compute quantiles (e.g., per action dimension).
        q_low (float): Lower quantile (default 1%).
        q_high (float): Upper quantile (default 99%).
    
    Returns:
        numpy array: Normalized data.
    """
    N, T, A = data.shape
    _data = data.reshape(N*T, A)
    q1 = np.percentile(_data, q_low, axis=axis, keepdims=True)
    q99 = np.percentile(_data, q_high, axis=axis, keepdims=True)
    # Compute scale factor and avoid division by zero
    scale = q99 - q1
    scale[scale < epsilon] = 1  # Prevent division by zero (maps all zeros to 0)
    
    # Apply linear transformation
    normalized_data = 2 * (_data - q1) / scale - 1

    # Clip values to ensure they stay within [-1, 1]
    normalized_data = np.clip(normalized_data, -1, 1)
    normalized_data = normalized_data.reshape(N, T, A)

    return normalized_data

class ParallelCosmosEncoder(torch.nn.Module):
    def __init__(self, encoder):
        super(ParallelCosmosEncoder, self).__init__()
        self.encoder = encoder.encoder
        self.quant_conv = encoder.quant_conv

    def forward(self, x):
        x = self.encoder(x)  # First pass through encoder
        x = self.quant_conv(x)  # Then pass through quantization convolution
        return x

ray.init(log_to_driver=False)
@ray.remote(num_cpus=64)
class ProcessTrajRay:
    def __init__(self, num_frames=8, chunk_length=20):
        # self.encoder = CausalVideoTokenizer(checkpoint_enc=f'{cosmos_path}/checkpoints/{model_name}/encoder.jit')
        # del self.model
        self.num_frames = num_frames
        self.chunk_length = chunk_length
    
    def prepare_input(self, np_array:np.ndarray) -> torch.Tensor: # (T, H, W, C)
        assert np_array.ndim == 4 # (T, H, W, C)
        # input_tensor = torch.from_numpy(np_array)
        duration = np_array.shape[0]
        frame_id_list = np.linspace(0, duration-1, self.num_frames, dtype=int)
        video_data = np_array[frame_id_list] #.permute(3, 0, 1, 2) # (T, H, W, C) -> (C, T, H, W)
        # permute the np array
        video_data = np.moveaxis(video_data, -1, 0)
        video_data = video_data / 255.0
        return video_data
    
    def process_sub_traj(self, traj, start, end):
        traj_len = traj['action'].shape[0]
        end = min(start + self.chunk_length, traj_len)
        frame_id_list = np.linspace(0, end-start-1, self.num_frames, dtype=int)
        sub_traj = {}
        for key in image_keys:
            sub_traj[key] = self.prepare_input(traj[key][start:end])
        for key in store_keys:
            if key == 'id':continue
            sub_traj[key] = traj[key][start:end]
            sub_traj[key] = sub_traj[key][frame_id_list]
            
        sub_traj['id'] = traj['id'] +'-'+str(start)+'-'+str(end)
        return sub_traj
    
class FeatureExtractor:
    def __init__(self, cosmos_path='Cosmos', model_name="Cosmos-0.1-Tokenizer-CV8x16x16", num_frames=8, chunk_length=20):
        cosmos_path = os.path.expanduser(cosmos_path)
        self.encoder = CausalVideoTokenizer(checkpoint_enc=os.path.join(cosmos_path, 'checkpoints', model_name, 'encoder.jit'))._enc_model
        self.encoder = ParallelCosmosEncoder(self.encoder)

        self.num_frames = num_frames
        self.ray_process = ProcessTrajRay.remote(num_frames, chunk_length)
        self.chunk_length = chunk_length
        self.data_queue = []
        self.traj_meta = {}
        self.no_more_data = False
        
        if torch.cuda.device_count() > 1:
            self.device = torch.device('cpu')
            self.num_gpu = torch.cuda.device_count()
            self.encoder = torch.nn.DataParallel(self.encoder, device_ids=range(self.num_gpu)).cuda()
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.encoder = self.encoder.to(self.device)
        self.encoder.eval()
    
    def get_features(self, torch_tensor: torch.Tensor) -> torch.Tensor: # (B, D)
        assert torch_tensor.ndim == 5, torch_tensor.ndim # (B, C, T, H, W)
        B = torch_tensor.shape[0]
        with torch.no_grad():
            image_embeds = self.encoder(torch_tensor.to(self.device))

        return image_embeds
    
    def fetch_data_fn(self, dataset_name, data_dir, split='train', batch_size=64):
        with tf.device('/CPU:0'):
            builder = tfds.builder(dataset_name, data_dir=data_dir)
            dataset = builder.as_dataset(split=split, decoders={"steps": tfds.decode.SkipDecoding()}, shuffle_files=False, read_config=tfds.ReadConfig(add_tfds_id=True))
            dataset_statistics = get_statistics(dataset_name, builder)
            
            obs_key = OXE_DATASET_CONFIGS[dataset_name]['image_obs_keys']['primary']
            if obs_key is None:
                obs_key = OXE_DATASET_CONFIGS[dataset_name]['image_obs_keys']['wrist']
            
            def get_dict(x):
                steps = OXE_STANDARDIZATION_TRANSFORMS[dataset_name](x['steps'])
                epsilon = 1e-8
                std = dataset_statistics['action']['std'][:-1]
                std_with_epsilon = np.where(std < epsilon, epsilon, std)
                xyzrpy = (steps['action'][:, :-1] - dataset_statistics['action']['mean'][:-1]) / std_with_epsilon
                normalized_action = tf.concat([xyzrpy, steps['action'][:, -1:]], axis=-1)
                    
                dicts = {
                    'image': tf.map_fn(tf.image.decode_jpeg, x['steps']['observation'][obs_key], dtype=tf.uint8),
                    'id': x['tfds_id'],
                    'action': steps['action'],
                    'normalized_action': normalized_action,
                    'proprio': steps['observation']['proprio']
                }
                return dicts
            dataset=dataset.map(get_dict, num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.prefetch(tf.data.AUTOTUNE)
            
            for i,traj in enumerate(tqdm(dataset)):
                for key in traj.keys():
                    traj[key] = traj[key].numpy()
                traj['id'] = traj['id'].decode()
                while len(self.data_queue) > batch_size * 2:
                    time.sleep(0.05)
                self.traj_meta[traj['id']] = {'sub_traj_id': [], 'sub_traj_len': []}
                traj_len = traj['action'].shape[0]
                for start in range(0, traj_len, self.chunk_length):
                    job = self.ray_process.process_sub_traj.remote(traj, start, start+self.chunk_length)
                    args = (traj, start, start+self.chunk_length)
                    self.traj_meta[traj['id']]['sub_traj_id'].append(traj['id'] +'-'+str(start)+'-'+str(start+self.chunk_length))
                    self.traj_meta[traj['id']]['sub_traj_len'].append(traj['action'].shape[0])
                    self.data_queue.append((job, args))
            
            self.no_more_data = True
                    
    def get_batch(self, batch_size):
        def str_find_second_last(string, substring):
            return string.rfind(substring, 0, string.rfind(substring))
        
        while len(self.data_queue) < batch_size and not self.no_more_data:
            time.sleep(0.02)
        if self.no_more_data and len(self.data_queue) < batch_size:
            batch_size = 1
            
        data_list = []
        while len(data_list) < batch_size:
            job, args = self.data_queue.pop(0)
            data = ray.get(job, timeout=15)
            data_list.append(data)
        
        batch = {}
        for key in data_list[0].keys():
            if key != 'id':
                batch[key] = np.stack([elem[key] for elem in data_list], axis=0)
            else:
                batch[key] = [elem[key] for elem in data_list]
                
        return batch

    def get_video_features_from_tfds(self, dataset_name, data_dir, split='train', batch_size=64):
        
        fetch_data_thread = Thread(target=self.fetch_data_fn, args=(dataset_name, data_dir, split, batch_size))
        fetch_data_thread.start()
            
        feature = dict()
        for key in store_keys:
            feature[key] = []
        feature['image_embeds'] = []
        
        while not self.no_more_data or len(self.data_queue) > 0:
            batch = self.get_batch(batch_size)
            while len(self.data_queue) < batch_size and not self.no_more_data:
                time.sleep(0.02)  # wait for more data on the queue to be processed
                
            input_tensor = torch.tensor(batch['image']).to(torch.bfloat16)
            output = self.get_features(input_tensor)
            video_features = output.cpu().to(torch.float32).numpy()
            
            for key in store_keys:
                feature[key].append(batch[key])
            feature['image_embeds'].append(video_features)
        
        assert self.no_more_data and len(self.data_queue) == 0, 'Data queue should be empty'
        
        for key in feature.keys():
            feature[key] = np.concatenate(feature[key], axis=0)
        return feature
    
def main(data_dir: str, 
         output_dir:str,
         dataset_name:str=None,
         split:str="train", 
         batch_size:int=64, 
         cosmos_path:str='Cosmos',
         model_name:str='Cosmos-0.1-Tokenizer-CV8x16x16',
         chunk_time:float=3.0,
         ):
    dataset_names = os.listdir(data_dir)
    if dataset_name is not None:
        assert dataset_name in dataset_names, f'{dataset_name} not in {dataset_names}'
        dataset_names = [dataset_name]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f'{output_dir}/config.json', 'w') as f:
        json.dump({'cosmos_path': cosmos_path, 'model_name': model_name, 'chunk_time': chunk_time}, f, indent=4)

    for dataset_name in dataset_names:
        if dataset_name not in OXE_DATASET_CONFIGS:
            continue
        print(f'Processing {dataset_name}')
        chunk_length = int(chunk_time * OXE_DATASET_CONTROL_FREQUENCY[dataset_name])
        feature_extractor = FeatureExtractor(cosmos_path, model_name, chunk_length=chunk_length)
        video_features = feature_extractor.get_video_features_from_tfds(dataset_name, data_dir, split, batch_size)
        output_dir = os.path.expanduser(output_dir)
        
        out_dir = f'{output_dir}/{dataset_name}'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        with open(f'{out_dir}/features.pkl', 'wb') as f:
            pickle.dump(video_features, f)
        with open(f'{out_dir}/traj_meta.pkl', 'wb') as f:
            pickle.dump(feature_extractor.traj_meta, f)
        print(f'Features saved to {out_dir}')
    
if __name__ == '__main__':
    tyro.cli(main)