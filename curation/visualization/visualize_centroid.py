from curation.utils.view_dataset_info import TrajLoader
import tyro
import numpy as np
import os
import pickle
from pathlib import Path
import natsort

def get_dataset_idx_range(metadata, i):
    sequence_length = [x[1] for x in metadata['sequence']]
    start_idx = sum(sequence_length[:i])
    end_idx = start_idx + sequence_length[i]
    return start_idx, end_idx

def main(data_dir:str, 
         centroid_file_dir:str, 
         exp_timestamp:str=None,
         centroids_to_visualize:list[int]=[0,1,2,3,4], 
         traj_per_cent:int=5,
         stride:int=1, 
         split:str="all"
         ):
    exps = os.listdir(centroid_file_dir)
    if exp_timestamp is not None:
        assert exp_timestamp in exps, f"exp_timestampe {exp_timestamp} not in {exps}"
        centroid_file_dir = os.path.join(centroid_file_dir, exp_timestamp)
    else:
        latest_exp = natsort.natsorted(exps)[-1]
        centroid_file_dir = os.path.join(centroid_file_dir, latest_exp)
        
    with open(os.path.join(centroid_file_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    centroid_array = np.load(os.path.join(centroid_file_dir, 'nearest_cent.npy'))
    dist_to_cent = np.load(os.path.join(centroid_file_dir, 'dist_to_cent.npy'))
    
    if centroid_file_dir[-1] == '/':
        centroid_file_base = os.path.basename(centroid_file_dir[:-1])
    else:
        centroid_file_base = os.path.basename(centroid_file_dir)
        
    cent_indices_dict = dict()
    for cent in centroids_to_visualize:
        cent_indices = np.argwhere(centroid_array==cent).squeeze()
        cent_indices = cent_indices.squeeze()
        cent_indices_dict[cent] = cent_indices
        
    for i, (dataset_name, dataset_length) in enumerate(metadata['sequence']):
        loader = TrajLoader(dataset_name, data_dir, split)  
        start_idx, end_idx = get_dataset_idx_range(metadata, i)
        for cent, cent_indices in cent_indices_dict.items():
            current_cent_indices = cent_indices[(cent_indices>=start_idx) & (cent_indices<end_idx)] - start_idx
            ids = metadata['id'][dataset_name][current_cent_indices]
            idx_id_to_visualize = zip(current_cent_indices[:traj_per_cent*stride:stride], ids[:traj_per_cent*stride:stride])
            print(f'\nVisualizing centroid {cent} from dataset {dataset_name}')
            loader.visualize(idx_id_to_visualize, save_dir=f'./video/{centroid_file_base}/cent_{cent}')
        
    
    
if __name__ == "__main__":
    tyro.cli(main)