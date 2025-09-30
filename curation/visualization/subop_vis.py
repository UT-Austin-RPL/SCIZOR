from curation.utils.view_dataset_info import TrajLoader
import tyro
import numpy as np
import os
import pickle
from curation.suboptimal_classifier.dataset.rlds.oxe.oxe_dataset_configs import OXE_DATASET_CONTROL_FREQUENCY, OXE_DATASET_CONFIGS

def window_average(scores, traj_meta, average_time, dataset_name):
    average_window = int(OXE_DATASET_CONTROL_FREQUENCY[dataset_name] * average_time)
    average_window = max(average_window, 1)
    assert len(scores.shape) == 1, f"Scores shape {scores.shape} is not 1D"
    pad_width = (average_window//2, average_window//2+1)
    
    from concurrent.futures import ThreadPoolExecutor
    def smooth_score(tfds_id):
        start, end = traj_meta[tfds_id]['start'], traj_meta[tfds_id]['end']
        cur_score = scores[start:end]
        pad_score = np.pad(cur_score, pad_width, mode='edge')
        smoothed_score = np.convolve(pad_score, np.ones(average_window)/average_window, mode='same')
        scores[start:end] = smoothed_score[pad_width[0]:-pad_width[1]]

    jobs = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        for tfds_id in traj_meta:
            jobs.append(executor.submit(smooth_score, tfds_id))
        
    for job in jobs:
        job.result()
        
    return scores

def main(data_dir:str, 
         subop_file_dir:str,
         small_percentile:float=0.975,
         big_percentile:float=1.0,
         min_subop_sec:float=0,
         dataset_name:str=None,
         save_dir:str='./video',
         average_time:float=2.0,
         num_per_dataset:int=5,
         ):
    dataset_names = os.listdir(data_dir)
    if dataset_name is not None:
        assert dataset_name in dataset_names, f"dataset_name {dataset_name} not in {dataset_names}"
        dataset_names = [dataset_name]
    
    for dataset_name in dataset_names:
        if dataset_name not in OXE_DATASET_CONFIGS or OXE_DATASET_CONFIGS[dataset_name]['image_obs_keys']['primary'] is None:
            print(f"Dataset {dataset_name} not in OXE_DATASET_CONFIG, skipping")
            continue
        traj_to_vis = []
        visualizer = TrajLoader(dataset_name, data_dir)
        score_file = os.path.join(subop_file_dir, dataset_name, 'scores.npy')
        metadata_file = os.path.join(subop_file_dir, dataset_name, 'traj_meta.pkl')
        scores = np.load(score_file)
        with open(metadata_file, 'rb') as f:
            traj_meta = pickle.load(f)['traj_meta']
            
        scores = window_average(scores.squeeze(), traj_meta, average_time, dataset_name)
        mask = scores > np.percentile(scores, small_percentile*100)
        mask = mask & (scores <= np.percentile(scores, big_percentile*100))
            
        for traj_id, info in traj_meta.items():
            cur_mask = mask[info['start'] : info['end']]
            if np.sum(cur_mask) < min_subop_sec * OXE_DATASET_CONTROL_FREQUENCY[dataset_name]:
                continue
            if np.any(cur_mask):
                traj_to_vis.append((traj_id, np.where(cur_mask)[0].tolist()))
            if len(traj_to_vis) >= num_per_dataset:
                break
                
        visualizer.visualize(traj_to_vis, save_dir=f'{save_dir}/{dataset_name}')
    
if __name__ == "__main__":
    tyro.cli(main)