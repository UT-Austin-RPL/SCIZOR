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
import time
from threading import Thread
from utils import get_statistics
import json
import h5py
from torchvision.transforms import functional as F

store_keys = ['id', 'action', 'normalized_action']

class Hdf5Dataset(torch.utils.data.Dataset):
    def __init__(self, name, file_path, num_frames, chunk_length, image_key, image_size, action_max_dim):
        self.name = name
        self.file = h5py.File(file_path, 'r')
        self.chunk_length = chunk_length
        self.image_size = image_size
        self.num_frames = num_frames
        self.action_max_dim = action_max_dim
        self.image_key = image_key
        self.keys = list(self.file.keys())
        self.total_len, self.idx_mapping = self.count_length()
        self.calculate_statistics()
        
    def pad_action(self, action):
        padding = np.zeros((len(action), self.action_max_dim - action.shape[-1]))
        action = np.concatenate([action, padding], axis=-1)
        return action
        
    def calculate_statistics(self):
        # action_dim = self.file['data'][list(self.file['data'].keys())[0]]['actions'].shape[-1]
        self.mean = np.zeros(self.action_max_dim)
        self.std = np.zeros(self.action_max_dim)
        self.count = 0
        for demo_key in self.file['data'].keys():
            action = self.file['data'][demo_key]['actions']
            action = self.pad_action(action)
            self.mean += np.sum(action, axis=0)
            self.count += action.shape[0]
            
        self.mean /= self.count
        
        for demo_key in self.file['data'].keys():
            action = self.file['data'][demo_key]['actions']
            action = self.pad_action(action)
            self.std += np.sum((action - self.mean)**2, axis=0)
        
        self.std = np.sqrt(self.std / self.count)
        
    def preprocess_image(self, video_data):
        video_data = torch.from_numpy(video_data)
        # duration = np_array.shape[0]
        frame_id_list = np.linspace(0, len(video_data)-1, self.num_frames, dtype=int)
        video_data = video_data[frame_id_list].numpy()
        video_data = np.moveaxis(video_data, -1, 0) # (T, H, W, C) -> (C, T, H, W)
        video_data = video_data / 255.0
        video_data = F.resize(torch.tensor(video_data), (self.image_size, self.image_size))
        return video_data
    
    def sub_sample_action(self, action):
        duration = action.shape[0]
        frame_id_list = np.linspace(0, duration-1, self.num_frames, dtype=int)
        return action[frame_id_list]
        
    def count_length(self):
        total_len = 0
        idx_mapping = []
        for demo_key in self.file['data'].keys():
            demo = self.file['data'][demo_key]
            traj_len = len(demo['actions'])
            for i in range(0, traj_len, self.chunk_length):
                idx_mapping.append({'demo_key':demo_key, 'start':i, 'end':min(i+self.chunk_length, traj_len), 'id':'-'.join([self.name, demo_key, str(i), str(min(i+self.chunk_length, traj_len))])})
                total_len += 1
        return total_len, idx_mapping
        
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx):
        idx_info = self.idx_mapping[idx]
        image = self.file[f"data/{idx_info['demo_key']}/obs/{self.image_key}"][idx_info['start']:idx_info['end']]
        action = self.file[f"data/{idx_info['demo_key']}/actions"][idx_info['start']:idx_info['end']]
        action = self.pad_action(action)
        normalized_action = (action - self.mean) / (self.std + 1e-6)
        data = {'image': image, 'action': action, 'normalized_action': normalized_action, 'id': idx_info['id']}
        data['image'] = self.preprocess_image(data['image'])
        data['action'] = self.sub_sample_action(data['action'])
        data['normalized_action'] = self.sub_sample_action(data['normalized_action'])
        return data


class ParallelCosmosEncoder(torch.nn.Module):
    def __init__(self, encoder):
        super(ParallelCosmosEncoder, self).__init__()
        self.encoder = encoder.encoder
        self.quant_conv = encoder.quant_conv

    def forward(self, x):
        x = self.encoder(x)  # First pass through encoder
        x = self.quant_conv(x)  # Then pass through quantization convolution
        return x

    
class FeatureExtractor:
    def __init__(self, cosmos_path='Cosmos', model_name="Cosmos-0.1-Tokenizer-CV8x16x16", num_frames=8, chunk_length=20):
        cosmos_path = os.path.expanduser(cosmos_path)
        self.encoder = CausalVideoTokenizer(checkpoint_enc=os.path.join(cosmos_path, 'checkpoints', model_name, 'encoder.jit'))._enc_model
        self.encoder = ParallelCosmosEncoder(self.encoder)

        self.num_frames = num_frames
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

    
    def get_dataloader(self, name, dataset_path, batch_size=64, image_key='agentview_image', image_size=128, action_max_dim=14):
        datasets = Hdf5Dataset(name ,dataset_path, num_frames=self.num_frames, chunk_length=self.chunk_length, image_key=image_key, image_size=image_size, action_max_dim=action_max_dim)
        dataloader = torch.utils.data.DataLoader(datasets, batch_size=batch_size, shuffle=False, num_workers=0)
        return dataloader
        

    def get_video_features_from_hdf5(self, name, dataset_path, batch_size=64, image_key='agentview_image', image_size=128, action_max_dim=14):
        data_loader = self.get_dataloader(name, dataset_path, batch_size, image_key, image_size, action_max_dim)
            
        feature = dict()
        for key in store_keys:
            feature[key] = []
        feature['image_embeds'] = []
        
        for batch in tqdm(data_loader):
            input_tensor = torch.tensor(batch['image']).to(torch.bfloat16)
            output = self.get_features(input_tensor)
            B = output.shape[0]
            video_features = output.cpu().to(torch.float32).numpy().reshape(B, -1)
            
            feature['image_embeds'].append(video_features)
            for key in store_keys:
                feature[key].append(batch[key])
        
        for key in feature.keys():
            feature[key] = np.concatenate(feature[key], axis=0)
        return feature
    
def main(data_dir: str, 
         output_dir:str,
         dataset_name:str=None,
         batch_size:int=64, 
         cosmos_path:str='Cosmos',
         model_name:str='Cosmos-0.1-Tokenizer-CV8x16x16',
         chunk_time:float=2.0,
         freq:int=20,
         image_key:str='agentview_image',
         image_size:int=128,
         action_max_dim:int=14
         ):
    #walk
    dataset_names = []
    dataset_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.hdf5'):
                if 'robomimic' in root:
                    dataset_names.append(root.split('/')[-2])
                else:
                    dataset_names.append(file.split('.')[0])
                dataset_paths.append(os.path.join(root, file))
                
    if dataset_name is not None:
        assert dataset_name in dataset_names, f'{dataset_name} not in {dataset_names}'
        dataset_names = [dataset_name]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(f'{output_dir}/config.json', 'w') as f:
        json.dump({'cosmos_path': cosmos_path, 'model_name': model_name, 'chunk_time': chunk_time}, f, indent=4)

    for i,dataset_path in enumerate(dataset_paths):
        print(f'Processing {dataset_names[i]}: {dataset_path}')
        chunk_length = int(chunk_time * freq)
        feature_extractor = FeatureExtractor(cosmos_path, model_name, chunk_length=chunk_length)
        video_features = feature_extractor.get_video_features_from_hdf5(dataset_names[i] ,dataset_path, batch_size, image_key, image_size, action_max_dim=action_max_dim)
        output_dir = os.path.expanduser(output_dir)
        
        out_dir = f'{output_dir}/{dataset_names[i]}'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        with open(f'{out_dir}/features.pkl', 'wb') as f:
            pickle.dump(video_features, f)
        print(f'Features saved to {out_dir}')
    
if __name__ == '__main__':
    tyro.cli(main)