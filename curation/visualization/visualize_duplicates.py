from curation.utils.view_dataset_info import TrajLoader
import tyro
import numpy as np
import os
import pickle
from pathlib import Path
import natsort
import tensorflow as tf

def get_dataset_idx_range(metadata, i):
    sequence_length = [x[1] for x in metadata['sequence']]
    start_idx = sum(sequence_length[:i])
    end_idx = start_idx + sequence_length[i]
    return start_idx, end_idx

def main(data_dir:str, 
         centroid_file_dir:str,
         exp_timestamp:str=None, 
         centroids_to_visualize:list[int]=[0,1,2,3,4,5,6,7,8,9], 
         split:str="all",
         eps_list:list[float]=[0.01, 0.02, 0.03, 0.05, 0.07, 0.1],
         max_group_per_cent:int=10,
         ):
    exps = os.listdir(centroid_file_dir)
    if exp_timestamp is not None:
        assert exp_timestamp in exps, f"exp_timestampe {exp_timestamp} not in {exps}"
        centroid_file_dir = os.path.join(centroid_file_dir, exp_timestamp)
    else:
        latest_exp = natsort.natsorted(exps)[-1]
        centroid_file_dir = os.path.join(centroid_file_dir, latest_exp)
        
    print(f"Visualizing duplicates from {centroid_file_dir}")

    with open(os.path.join(centroid_file_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    centroid_semdedup_folder = Path(centroid_file_dir, 'semdedup', 'dataframes')
    
    if centroid_file_dir[-1] == '/':
        centroid_file_base = os.path.basename(centroid_file_dir[:-1])
    else:
        centroid_file_base = os.path.basename(centroid_file_dir)
    
    
    idx_to_visualize = []
    tfds_id_to_visualize = []
    centroids = []
    eps_arr = []
    duplicate_group = dict()
    group_cnt = 0
    for cent in centroids_to_visualize:
        semdedup_path = Path(centroid_semdedup_folder, f'cluster_{cent}.pkl')
        with open(semdedup_path, 'rb') as f:
            cluster = pickle.load(f)
            
        for num_eps, eps in enumerate(eps_list):
            assert f"eps={eps}" in cluster, f"eps={eps} not in cluster"
            prev_eps = eps_list[num_eps-1] if num_eps > 0 else None
            cur_duplicates = cluster[f"eps={eps}"]
            if prev_eps is not None:
                prev_duplicates = cluster[f"eps={prev_eps}"]
                is_duplicate = cur_duplicates & (~prev_duplicates)
            else:
                is_duplicate = cur_duplicates
                
            duplicates = cluster[is_duplicate][:max_group_per_cent]
            duplicates_idx = duplicates['id_in_dataset'].to_numpy() # idx is the id in the dataset
            duplicate_tfds_id = duplicates['tfds_id'].to_numpy()
            most_similar_cluster_id = duplicates['most_similar_idx']
            most_similar = cluster.iloc[most_similar_cluster_id]
            most_similar_idx = most_similar['id_in_dataset'].to_numpy()
            most_similar_tfds_id = most_similar['tfds_id'].to_numpy()
            print(f'Centroid {cent} has {cur_duplicates.sum()}/{len(cluster)} duplicates for eps<{eps}, {is_duplicate.sum()}/{len(cluster)} duplicates for {prev_eps}<eps<{eps}')
            idx_to_visualize += [duplicates_idx, most_similar_idx]
            tfds_id_to_visualize += [duplicate_tfds_id, most_similar_tfds_id]
            centroids += [np.full_like(duplicates_idx, cent), np.full_like(most_similar_idx, cent)]
            eps_arr += [np.full_like(duplicates_idx, eps, dtype=np.float32), np.full_like(most_similar_idx, eps, dtype=np.float32)]
            def group_append(tfds_id, group):
                if tfds_id not in duplicate_group:
                    duplicate_group[tfds_id] = group
                else:
                    duplicate_group[tfds_id].extend(group)
            for i in range(len(duplicate_tfds_id)):
                if duplicate_tfds_id[i] not in duplicate_group and most_similar_tfds_id[i] not in duplicate_group:
                    # duplicate_group[duplicate_tfds_id[i]] = group_cnt
                    # duplicate_group[most_similar_tfds_id[i]] = group_cnt
                    group_append(duplicate_tfds_id[i], [group_cnt])
                    group_append(most_similar_tfds_id[i], [group_cnt])
                    group_cnt += 1
                elif duplicate_tfds_id[i] in duplicate_group:
                    # duplicate_group[most_similar_tfds_id[i]] = duplicate_group[duplicate_tfds_id[i]]
                    group_append(most_similar_tfds_id[i], duplicate_group[duplicate_tfds_id[i]])
                elif most_similar_tfds_id[i] in duplicate_group:
                    # duplicate_group[duplicate_tfds_id[i]] = duplicate_group[most_similar_tfds_id[i]]
                    group_append(duplicate_tfds_id[i], duplicate_group[most_similar_tfds_id[i]])
                else: # both in duplicate_group
                    group_append(most_similar_tfds_id[i], duplicate_group[duplicate_tfds_id[i]])
                    group_append(duplicate_tfds_id[i], duplicate_group[duplicate_tfds_id[i]])
    print(f'Total {group_cnt} groups of duplicates.')
        
    idx_to_visualize = np.concatenate(idx_to_visualize, axis=0)
    tfds_id_to_visualize = np.concatenate(tfds_id_to_visualize, axis=0)
    centroids = np.concatenate(centroids, axis=0)
    eps_arr = np.concatenate(eps_arr, axis=0)
    # breakpoint()
    
    
        
    with tf.device('/CPU:0'):
        for i, (dataset_name, dataset_length) in enumerate(metadata['sequence']):
            loader = TrajLoader(dataset_name, data_dir, split)  
            start_idx, end_idx = get_dataset_idx_range(metadata, i)
            current_indices = (idx_to_visualize>=start_idx) & (idx_to_visualize<end_idx)
            current_idx = idx_to_visualize[current_indices] - start_idx
            _, unique_indices = np.unique(current_idx, return_index=True)
            
            current_cent = centroids[current_indices]
            current_tfds_id = tfds_id_to_visualize[current_indices]
            
            tfds_id_list, frames_list = [], []
            current_tfds_id_unique = current_tfds_id[unique_indices]
            for chunk_id in current_tfds_id_unique:
                sep = chunk_id.rfind('-', 0, chunk_id.rfind('-'))
                tfds_id = chunk_id[:sep]
                start, end = chunk_id[sep+1:].split('-')
                start, end = int(start), int(end)
                frames = list(range(start, end))
                tfds_id_list.append(tfds_id)
                frames_list.append(frames)
                
            current_eps = eps_arr[current_indices]
            
            idx_id_to_visualize = zip(current_idx[unique_indices], tfds_id_list, frames_list)
            loader.visualize(idx_id_to_visualize, save_dir=f'./video/{centroid_file_base}/duplicates')
            
            for cent, tfds_id, eps in zip(current_cent, current_tfds_id, current_eps):
                current_group = duplicate_group[tfds_id]
                sep = tfds_id.rfind('-', 0, tfds_id.rfind('-'))
                video_name = tfds_id[:sep]
                base_folder = f"./video/{centroid_file_base}/duplicates"
                for group in current_group:
                    current_folder = f"{base_folder}/eps_{eps:.2f}/cent_{cent}/group_{group}"
                    if not os.path.exists(current_folder):
                        os.makedirs(current_folder)
                    os.system(f'cp {base_folder}/{video_name}.mp4 {current_folder}/{video_name}.mp4')
            os.system(f'rm ./video/{centroid_file_base}/duplicates/*.mp4')

    
if __name__ == "__main__":
    tyro.cli(main)