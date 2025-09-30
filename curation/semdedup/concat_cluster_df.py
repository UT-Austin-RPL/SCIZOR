import os
import natsort
from pathlib import Path
import pickle
import tyro
import numpy as np
import pprint
from tqdm import tqdm
import pandas as pd
import argparse
import yaml

def main(centroid_file_dir:str,
         exp_timestamp:str=None, 
         ):
    exps = os.listdir(centroid_file_dir)
    if exp_timestamp is not None:
        assert exp_timestamp in exps, f"exp_timestampe {exp_timestamp} not in {exps}"
        centroid_file_dir = os.path.join(centroid_file_dir, exp_timestamp)
    else:
        latest_exp = natsort.natsorted(exps)[-1]
        centroid_file_dir = os.path.join(centroid_file_dir, latest_exp)
        
    print(f"Sumarizing from {centroid_file_dir}")
    
    with open(os.path.join(centroid_file_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    centroid_semdedup_folder = Path(centroid_file_dir, 'semdedup', 'dataframes')
    centroids = os.listdir(centroid_semdedup_folder)
    
    with open(os.path.join(centroid_semdedup_folder, 'cluster_0.pkl'), 'rb') as f:
        example_cent = pickle.load(f)
    
    summary = dict()
    concat_df = []
    for dataset in metadata['sequence']:
        dataset_name, num_traj = dataset
        summary[dataset_name] = dict()
        summary[dataset_name]['num_traj'] = num_traj
        summary[dataset_name]['duplicate'] = dict()
        for colunm in example_cent.columns:
            if 'eps' in colunm:
                summary[dataset_name]['duplicate'][colunm] = dict()
                summary[dataset_name]['duplicate'][colunm]['sum'] = 0
                summary[dataset_name]['duplicate'][colunm]['percentage'] = 0
        
    for cent in tqdm(centroids):
        if cent == 'concat_df.pkl':
            continue
        semdedup_path = Path(centroid_semdedup_folder, cent)
        with open(semdedup_path, 'rb') as f:
            cluster = pickle.load(f)
            cluster['tfds_id'] = cluster['tfds_id'].astype(str)
            tfds_id = cluster['tfds_id'].to_numpy().astype(str)
            id_index_cluster = cluster.set_index('tfds_id')
            concat_df.append(id_index_cluster)
            for dataset_name in summary.keys():
                tfds_id_in_dataset = np.char.count(tfds_id, dataset_name).astype(bool)
                corresponding_item = cluster[tfds_id_in_dataset]
                for colunm in summary[dataset_name]['duplicate'].keys():
                    summary[dataset_name]['duplicate'][colunm]['sum'] += len(corresponding_item[corresponding_item[colunm]])
                
    for dataset_name in summary.keys():
        for colunm in summary[dataset_name]['duplicate'].keys():
            summary[dataset_name]['duplicate'][colunm]['percentage'] = summary[dataset_name]['duplicate'][colunm]['sum']/summary[dataset_name]['num_traj']
            summary[dataset_name]['duplicate'][colunm]['percentage'] = round(summary[dataset_name]['duplicate'][colunm]['percentage'], 3)
    # make the summary dict looks better and print it 
    pprint.pprint(summary)
    print()
    total_num_traj = sum([x['num_traj'] for x in summary.values()])
    total_num_duplicate = dict()
    for colunm in example_cent.columns:
        if 'eps' in colunm:
            num_dup = sum([x['duplicate'][colunm]['sum'] for x in summary.values()])
            total_num_duplicate[colunm] = {'sum':num_dup, 'percentage':num_dup/total_num_traj}
    total_summary = {'total_num_traj':total_num_traj, 'total_num_duplicate':total_num_duplicate}
    pprint.pprint(total_summary)
    print("Concatenating dataframes")
    concat_df = pd.concat(concat_df)
    with open(os.path.join(centroid_semdedup_folder, 'concat_df.pkl'), 'wb') as f:
        pickle.dump(concat_df, f)
    print(f"Dataframes concatenated and saved to {centroid_semdedup_folder}/concat_df.pkl")
        
        
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config-file', type=str)
    arg = args.parse_args()
    with open(arg.config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    centroid_file_dir = config['save_folder']
    timestamp = config['timestamp']
    main(centroid_file_dir, timestamp)
    