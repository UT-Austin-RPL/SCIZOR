import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import dlimp as dl
from curation.utils.oxe_dataset_configs import OXE_DATASET_CONFIGS
from curation.utils.oxe_standardization_transforms import OXE_STANDARDIZATION_TRANSFORMS
from tqdm import tqdm
import tyro
import os
import pickle
from threading import Thread
from utils import get_statistics


def normalize_action(action, dataset_statistics):
    xyzrpy = (action[..., :-1] - dataset_statistics['action']['mean'][:-1][None,None,:]) / (dataset_statistics['action']['std'][:-1][None,None,:] + 1e-7)
    normalized_action = np.concatenate([xyzrpy, action[..., -1:]], axis=-1)
    return normalized_action

def main(data_dir: str, 
         input_dir:str,
         output_dir:str,
         ):
    dataset_names = os.listdir(data_dir)
    for dataset_name in dataset_names:
        if dataset_name not in OXE_DATASET_CONFIGS:
            continue
        print(f'Processing {dataset_name}')
        in_dir = f'{input_dir}/{dataset_name}'
        out_dir = f'{output_dir}/{dataset_name}'
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
            
        with open(f'{in_dir}/features.pkl', 'rb') as f:
            video_features = pickle.load(f)
        with open(f'{in_dir}/traj_meta.pkl', 'rb') as f:
            traj_meta = pickle.load(f)
        
        builder = tfds.builder(dataset_name, data_dir=data_dir)
        dataset_statistics = get_statistics(dataset_name, builder)
        
        video_features['normalized_action'] = normalize_action(video_features['action'], dataset_statistics)
            
        with open(f'{out_dir}/features.pkl', 'wb') as f:
            pickle.dump(video_features, f)
        with open(f'{out_dir}/traj_meta.pkl', 'wb') as f:
            pickle.dump(traj_meta, f)
        print(f'Features saved to {out_dir}')
    
if __name__ == '__main__':
    tyro.cli(main)