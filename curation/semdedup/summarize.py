import os
import natsort
from pathlib import Path
import pickle
import tyro
import numpy as np
import pprint
from tqdm import tqdm
import pandas as pd

def main(centroid_file_dir:str,
         ):
    exps = os.listdir(centroid_file_dir)
        
    print(f"Sumarizing from {centroid_file_dir}")
    
    with open(os.path.join(centroid_file_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    centroid_semdedup_folder = Path(centroid_file_dir, 'semdedup', 'dataframes')
    centroids = os.listdir(centroid_semdedup_folder)
    
    with open(os.path.join(centroid_semdedup_folder, 'concat_df.pkl'), 'rb') as f:
        concat_df = pickle.load(f)
        
    eps_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    concat_df['max_sim'] = np.nan_to_num(concat_df['max_sim'])
    thres_list = [np.percentile(concat_df['max_sim'], 100.0*(1-eps)) for eps in eps_list]
    print(thres_list)
    # breakpoint()
    summary = dict()
    for dataset in metadata['sequence']:
        dataset_name, num_traj = dataset
        summary[dataset_name] = dict()
        summary[dataset_name]['num_traj'] = num_traj
        summary[dataset_name]['duplicate'] = dict()
        for eps in eps_list:
            colunm = f'eps_{eps}'
            summary[dataset_name]['duplicate'][colunm] = dict()
            summary[dataset_name]['duplicate'][colunm]['sum'] = 0
            summary[dataset_name]['duplicate'][colunm]['percentage'] = 0
        
    concat_df['tfds_id'] = concat_df.index.astype(str)
    tfds_id = concat_df['tfds_id'].to_numpy().astype(str)
    for dataset_name in summary.keys():
        tfds_id_in_dataset = np.char.count(tfds_id, dataset_name).astype(bool)
        corresponding_item = concat_df[tfds_id_in_dataset]
        for i, colunm in enumerate(summary[dataset_name]['duplicate'].keys()):
            # summary[dataset_name]['duplicate'][colunm]['sum'] += len(corresponding_item[corresponding_item[colunm]])
            summary[dataset_name]['duplicate'][colunm]['sum'] += sum(corresponding_item['max_sim'] > thres_list[i])
                
    for dataset_name in summary.keys():
        for colunm in summary[dataset_name]['duplicate'].keys():
            summary[dataset_name]['duplicate'][colunm]['percentage'] = summary[dataset_name]['duplicate'][colunm]['sum']/summary[dataset_name]['num_traj']
            summary[dataset_name]['duplicate'][colunm]['percentage'] = round(summary[dataset_name]['duplicate'][colunm]['percentage'], 3)
    # make the summary dict looks better and print it 
    pprint.pprint(summary)
    print()
    total_num_traj = sum([x['num_traj'] for x in summary.values()])
    total_num_duplicate = dict()
    for colunm in summary[dataset_name]['duplicate'].keys():
        if 'eps' in colunm:
            num_dup = sum([x['duplicate'][colunm]['sum'] for x in summary.values()])
            total_num_duplicate[colunm] = {'sum':num_dup, 'percentage':num_dup/total_num_traj}
    total_summary = {'total_num_traj':total_num_traj, 'total_num_duplicate':total_num_duplicate}
    pprint.pprint(total_summary)
            
    
        
        
if __name__ == '__main__':
    tyro.cli(main)