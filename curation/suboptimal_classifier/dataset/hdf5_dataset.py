import os
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from curation.suboptimal_classifier.utils import process_char_array_for_dino, normalize_images
import bisect
import natsort
from torchvision.transforms import Resize


class Hdf5SubDataset(Dataset):
    def __init__(self, hdf5, config):
        self.hdf5 = hdf5
        self.total_demo_length = 0
        self.obs_keys = config.hdf5_dataset_kwargs['obs_keys']
        self.time_bins = config.hdf5_dataset_kwargs['time_bins']
        self.freq = config.hdf5_dataset_kwargs['freq']
        self.resize = {key: Resize((self.obs_keys[key], self.obs_keys[key])) for key in self.obs_keys}
        self.num_bins = len(self.time_bins)
        self.cumulative_demo_lengths = []
        
        for key in natsort.natsorted(self.hdf5['data'].keys()):
            demo = self.hdf5[f'data/{key}']
            if 'num_samples' in demo.attrs:
                length = demo.attrs['num_samples']
            else:
                length = len(demo['actions'])
            self.total_demo_length += length
            self.cumulative_demo_lengths.append(self.total_demo_length)
            
        
    def __len__(self):
        return self.total_demo_length
    
    def get_rand_dist(self, traj_len):
        rand_bins = np.random.randint(0, self.num_bins)
        cur_bin = self.time_bins[rand_bins]
        alpha = np.random.rand()
        rand_time = cur_bin[0] * alpha + min(cur_bin[1], traj_len/self.freq) * (1-alpha)
        rand_dist = np.round(rand_time * self.freq).astype(int)
        start_idx = np.random.randint(0, max(1, traj_len-1-rand_dist))
        end_idx = min(start_idx + rand_dist, traj_len-1)
        dist = end_idx - start_idx
        return start_idx, end_idx, dist
    
    def __getitem__(self, idx):
        demo_idx = bisect.bisect_right(self.cumulative_demo_lengths, idx)
        traj_len = self.cumulative_demo_lengths[demo_idx] if demo_idx == 0 else self.cumulative_demo_lengths[demo_idx] - self.cumulative_demo_lengths[demo_idx-1]
        assert traj_len == len(self.hdf5[f'data/demo_{demo_idx}/actions'])
        cur_idx, end_idx, dist = self.get_rand_dist(traj_len)
        score = delta_time = (end_idx - cur_idx) / self.freq
            
        image = {}
        for key in self.obs_keys:
            current_image = torch.tensor(self.hdf5[f'data/demo_{demo_idx}/obs/{key}'][cur_idx])
            future_image = torch.tensor(self.hdf5[f'data/demo_{demo_idx}/obs/{key}'][end_idx])
            image[key] = torch.stack([current_image, future_image], dim=0)
            image[key] = normalize_images(image[key], batched=False)
            image[key] = image[key].permute(0, 3, 1, 2)
            image[key] = self.resize[key](image[key])
            
        return image, score
    
class InterleaveDatasets(Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_lengths = [len(datasets[0])]
        for dataset in datasets[1:]:
            self.cumulative_lengths.append(len(dataset) + self.cumulative_lengths[-1])
    
    def __len__(self):
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, idx):
        for i, cum_length in enumerate(self.cumulative_lengths):
            if idx < cum_length:
                if i == 0:
                    idx_in_dataset = idx
                else:
                    idx_in_dataset = idx - self.cumulative_lengths[i-1]
                return self.datasets[i][idx_in_dataset]
        raise ValueError("Index out of bounds")
            

class HDF5Dataset():
    def __init__(self, config):
        assert config.future_image, "HDF5Dataset only implemented with future_image"
        self.hdf5_config = config.hdf5_dataset_kwargs
        self.image_key = config.discriminator_dataset_kwargs['image_key']
        self.paths = []
        for root, dirs, files in os.walk(self.hdf5_config['data_dir']):
            for file in files:
                if file.endswith('.hdf5'):
                    self.paths.append(os.path.join(root, file))
                    
        self.sub_datasets = {}
        for path in self.paths:
            hdf5_file = h5py.File(path, 'r')
            self.sub_datasets[path] = Hdf5SubDataset(hdf5_file, config)
            
        self.dataset = InterleaveDatasets(list(self.sub_datasets.values()))
        self.dataloader = DataLoader(self.dataset, batch_size=self.hdf5_config['batch_size'], shuffle=True, num_workers=self.hdf5_config['num_workers'], persistent_workers=True, prefetch_factor=2)
        # self.dataloader = DataLoader(self.dataset, batch_size=self.hdf5_config['batch_size'], shuffle=True, num_workers=self.hdf5_config['num_workers'])
        self.data_iter = iter(self.dataloader)
        
    def __len__(self):
        return len(self.dataloader)
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __next__(self):
        # if the iterator is exhausted, create a new one
        try:
            image, score = next(self.data_iter)
        except StopIteration:
            self.data_iter = iter(self.dataloader)
            image, score = next(self.data_iter)
        image = image[self.image_key]
        return image, None, score, None, None, None, None # match the legacy format
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    
        
            
        