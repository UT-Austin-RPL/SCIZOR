from natsort import natsorted
import h5py
import json
import tyro
import os
import torch
import numpy as np
from tqdm import tqdm
import imageio
import pickle

# only support latest image only discriminator
def write_max_sim(data_dir, semdedup_path):
    
    file_paths = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.hdf5'):
                file_paths.append(os.path.join(root, file))  
    
    assert semdedup_path.endswith('.pkl'), "semdedup_path should be a .pkl file"
    with open(semdedup_path, 'rb') as f:
        dedup_df = pickle.load(f)
    
    for file_path in file_paths:
        print("Processing", file_path)
        if os.path.basename(file_path) == 'image.hdf5': # robomimic dataset
            name = file_path.split('/')[-3]
        else:
            name = os.path.basename(file_path).split('.')[0]
            
        file = h5py.File(file_path, 'a')
        current_df =  dedup_df[np.char.count(dedup_df.index.to_numpy().astype(str), name)>0]
        for i, demo_key in tqdm(enumerate(natsorted(file['data'].keys())), total=len(file['data'].keys())):
            demo = file[f'data/{demo_key}']
            current_demo_df = current_df[np.char.count(current_df.index.to_numpy().astype(str), demo_key + '-')>0]
            if len(current_demo_df) == 0:
                print(f"Skipping {demo_key}")
                continue
            max_sim = np.zeros(len(demo['actions']))
            max_sim = []
            max_sim_idx = []
            for j in range(len(current_demo_df)):
                idx = current_demo_df.index[j]
                start_idx, end_idx = idx.split('-')[-2:]
                start_idx, end_idx = int(start_idx), int(end_idx)
                # max_sim[start_idx:end_idx] = current_demo_df.loc[idx, 'max_sim']
                max_sim.append(current_demo_df.loc[idx, 'max_sim'])
                max_sim_idx.append([start_idx, end_idx])
            max_sim = np.array(max_sim)
            max_sim_idx = np.array(max_sim_idx)
            print(max_sim.max(), max_sim.min())
            if "max_sim" in demo.keys():
                del demo['max_sim']
            if "max_sim_idx" in demo.keys():
                del demo['max_sim_idx']
            demo.create_dataset('max_sim', data=max_sim)
            demo.create_dataset('max_sim_idx', data=max_sim_idx)
        
        file.close()
                    
def main(
    data_dir: str, 
    semdedup_path: str,
):          
    write_max_sim(data_dir, semdedup_path)
                    
if __name__ == "__main__":
    tyro.cli(main)