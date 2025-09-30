# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import numpy as np
import pandas as pd
import submitit
import torch
from tqdm import tqdm
import pickle
import random
import math
import time
import pprint
import yaml
import natsort
from curation.semadedup.loader import EmbeddingLoader
import pickle
import pathlib


def init_memmap_embs(
    embs_memory_loc: str, dataset_size: int, emd_size: int = 512, dtype: str = "float32"
) -> np.memmap:
    """
    Initializes a memory-mapped NumPy array to read embeddings of examples.

    Args:
        embs_memory_loc (str): Path to the memory-mapped file.
        dataset_size (int): Size of the dataset.
        emd_size (int): Dimensionality of the embeddings.
        dtype (str): Data type of the embeddings.

    Returns:
        np.memmap: A memory-mapped NumPy array.
    """
    embs = np.memmap(
        embs_memory_loc, dtype=dtype, mode="r", shape=(dataset_size, emd_size)
    )
    return embs


class SemDeDupJob(submitit.helpers.Checkpointable):
    """
    - Each SLURMJob will run SemDeDup on number of clusters and save dataframe with which examples to keep from each cluster.
    - Parallelize job_start_cluster across jobs so that preemption in the middle of an epoch isn't a problem and because we want to
    keep the shard structure anyway.
    - Process more than one cluster per job=> run multiple taks inside each jobs.
    - Preempted jobs get resubmitted. Already precessed clusters get skipped internally.
    """

    def __init__(self, args, job_start_cluster: int):
        self.args = args
        self.job_start_cluster = job_start_cluster
        random.seed(args.seed)

    def _contains_duplicates(self, arr):
        return len(np.unique(arr)) != len(arr)

    def semdedup(self, cluster, cluster_reps, device):
        st = time.time()
        ## -- compute pairwise cos sim between cluster items, then replace to diagonal with zeros to ignore self similarity
        cluster_reps.to(device)
        if self.args.sim_metric == "cosine":
            pair_w_sim_matrix = cluster_reps @ (cluster_reps.T)
        elif self.args.sim_metric == "l2":
            pair_w_sim_matrix = torch.cdist(cluster_reps, cluster_reps, p=2)
            pair_w_sim_matrix = -pair_w_sim_matrix
        del cluster_reps
        pair_w_sim_matrix.fill_diagonal_(0.0)
        assert pair_w_sim_matrix.shape[0] == pair_w_sim_matrix.shape[1]

        ## -- get paths to cluster i images
        image_urls = cluster[:, 0]

        ## -- make sure all the paths are unique this ensure that the duplicates are really stored many time times on memory
        assert not self._contains_duplicates(image_urls)

        ## -- We need upper tringular matrix because (1)we don't need to look at self sim (always=1) (2)we need the compinations not permutations
        triu_sim_mat = torch.triu(pair_w_sim_matrix, diagonal=1)

        ## -- if the max sim between one example and any other example is > 1-eps, remove this example
        if not self.args.Kmeans_with_cos_dist:
            triu_sim_mat = torch.where(triu_sim_mat == 0, -torch.inf, triu_sim_mat)
        M, idx = torch.max(triu_sim_mat, dim=0)[0].cpu(), torch.argmax(triu_sim_mat, dim=0).cpu()
        print(f"Step time: {time.time()-st}(s)")

        return M, idx

    def _process_shard(self, start_cluster: int, end_cluster: int):
        # print("SemDeDup params: ", self.args)
        st = time.time()

        metadata_path = pathlib.Path(self.args.save_folder, "metadata.pkl")
        with open(metadata_path, "rb") as f:
            loaded_metadata = pickle.load(f)

        loader = EmbeddingLoader(self.args.emb_memory_folder, loaded_metadata["modalities"])
        normalization = self.args.Kmeans_with_cos_dist
        embs, metadata = loader.load_embeddings(normalization=normalization)
        assert metadata['sequence'] == loaded_metadata['sequence'], "Metadata Mismatch, the embedding components are not the same"
        

        step_time = []

        for cluster_id in tqdm(range(start_cluster, end_cluster)):
            step_st = time.time()
            
            save_folder = pathlib.Path(self.args.save_loc, "dataframes")
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            
            df_file_loc = os.path.join(
                self.args.save_loc, f"dataframes/cluster_{cluster_id}.pkl"
            )

            if os.path.exists(df_file_loc):  # and os.path.exists(dict_file_loc):
                print(f"{df_file_loc} exists, moving on")
                continue

            ## -- load cluster i representations
            cluster_i = np.load(
                os.path.join(
                    self.args.sorted_clusters_path, f"cluster_{cluster_id}.npy"
                )
            )
            # 1) store cluster size
            cluster_size = cluster_i.shape[0]
            print("cluster_size: ", cluster_size)

            if cluster_size == 1:
                points_to_remove_df = pd.DataFrame()
                points_to_remove_df["indices"] = [0]
                points_to_remove_df["tfds_id"] = cluster_i[:, 0]
                points_to_remove_df["id_in_dataset"] = cluster_i[:, 1].astype("int32")
                points_to_remove_df["most_similar_idx"] = [0]
                for eps in self.args.eps_list:
                    ## We need to remove a point from the dataset when its pairwise similarity to other point is > 1-ebs
                    points_to_remove_df[f"eps={eps}"] = [False]
                if self.args.save_loc != "":
                    ## --save df
                    with open(df_file_loc, "wb") as file:
                        pickle.dump(points_to_remove_df, file)
                print("DONE cluster_id ", cluster_id)
                continue

            ## -- By default, we keep hard examples from groups
            clutser_items_indices = list(range(cluster_size))
            ## -- OR: shuffle cluster to keep random example from each group
            if self.args.which_to_keep.lower() == "random":
                random.shuffle(clutser_items_indices)
                cluster_i = cluster_i[clutser_items_indices]
            ## -- OR: reverse cluster to keep easy examples
            if self.args.which_to_keep.lower() == "easy":
                clutser_items_indices = clutser_items_indices[::-1]
                cluster_i = cluster_i[clutser_items_indices]

            ## -- indices for cluster items in the dataset
            cluster_ids = cluster_i[:, 1].astype("int32")
            cluster_reps = embs[cluster_ids]
            cluster_reps = torch.tensor(cluster_reps)

            M, idx = self.semdedup(cluster_i, cluster_reps, self.args.device)

            points_to_remove_df = pd.DataFrame()
            points_to_remove_df["indices"] = clutser_items_indices
            points_to_remove_df["tfds_id"] = cluster_i[:, 0]
            points_to_remove_df["id_in_dataset"] = cluster_ids
            points_to_remove_df["most_similar_idx"] = idx.numpy()
            points_to_remove_df["max_sim"] = M.numpy()

            for eps in self.args.eps_list:
                ## -- 5) We need to remove a point from the dataset when its pairwise similarity to other point is > 1-ebs
                eps_points_to_remove = M > np.percentile(M, 100 * (1 - eps))
                points_to_remove_df[f"eps={eps}"] = eps_points_to_remove

            if self.args.save_loc != "":
                ## --save df
                with open(df_file_loc, "wb") as file:
                    pickle.dump(points_to_remove_df, file)

            step_time.append(time.time() - step_st)
            print("DONE cluster: ", cluster_id)

        print(
            f"DONE in {((time.time()-st)/60):.2f} minutes, Average Step time {(sum(step_time)/len(step_time)):.2f}(s)"
        )
        return

    def __call__(self):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(self.args))
        job_start_cluster = self.job_start_cluster

        print(
            f"This job will process clusters {job_start_cluster} to  {min(self.args.num_clusters, job_start_cluster+self.args.clusters_per_job)}"
        )

        job_env = submitit.JobEnvironment()

        print(f"There are {job_env.num_tasks} tasks in this job")

        print(f"I'm the task #{job_env.local_rank} on node {job_env.node}")
        print(f"I'm the task #{job_env.global_rank} in the job")
        self.args.device = "cuda:" + str(job_env.global_rank % torch.cuda.device_count())

        ## divide clusters across tasks (cpus)
        num_clusters_per_task = int(
            math.ceil(self.args.clusters_per_job / job_env.num_tasks)
        )
        task_rank = job_env.local_rank
        start_cluster = job_start_cluster + task_rank * num_clusters_per_task
        end_cluster = job_start_cluster + (task_rank + 1) * num_clusters_per_task
        end_cluster = min(self.args.num_clusters, end_cluster)
        end_cluster = min(end_cluster, job_start_cluster + self.args.clusters_per_job)
        print(
            f"This task will process {num_clusters_per_task} clusters: cluster {start_cluster} to cluster {end_cluster}"
        )
        print(
            f"This task will process cluster {start_cluster} to cluster {end_cluster}"
        )

        self._process_shard(start_cluster, end_cluster)
        
def launch(args):
    confg_file = args.config_file
    ## -- load kmeans clustering parameters from configs file
    with open(confg_file, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
    # with open(pathlib.Path(params["save_folder"], "clustering_params.txt"), "w") as f:
    #     pprint.pprint(params, f)

    args.save_folder = params["save_folder"]
    args.emb_memory_folder = params["emb_memory_folder"]
    args.timestamp = params.get("timestamp", None)
    args.num_clusters = params["ncentroids"]
    args.seed = params["seed"]
    args.sim_metric = params["sim_metric"]
    args.keep_hard = params["keep_hard"]
    args.Kmeans_with_cos_dist = params["Kmeans_with_cos_dist"]
    args.save_folder = params["save_folder"]
    args.modality = params["modality"]
    args.which_to_keep = params["which_to_keep"]
    args.eps_list = params["eps_list"]
    args.clusters_per_job = args.num_clusters // args.num_tasks + 1
    
    embeddings = natsort.natsorted(os.listdir(args.save_folder))
    if args.timestamp and args.timestamp in embeddings:
        args.save_folder = os.path.join(args.save_folder, args.timestamp)
    else:
        args.save_folder = os.path.join(args.save_folder, embeddings[-1])
        
    args.sorted_clusters_path = os.path.join(args.save_folder, "sorted_clusters")
    args.save_loc = os.path.join(args.save_folder, "semdedup")
    
    
        
        

    ## -- SLURM CONFIG
    PARTITION = args.partition
    SLURM_ARRAY_PARALLELISM = 1000
    NODES = 1
    TIMEOUT = args.timeout
    CPUS_PER_TASKS = args.cpus_per_task
    TASKS_PER_NODE = args.num_tasks
    NGPUS = args.ngpus

    ## -- SUBMIT
    submitit_path = f"{args.save_folder}/semdedup-jobs/%j"
    executor = submitit.AutoExecutor(folder=submitit_path, slurm_max_num_timeout=30)
    executor.update_parameters(
        slurm_partition=PARTITION,
        slurm_array_parallelism=SLURM_ARRAY_PARALLELISM,
        nodes=NODES,
        tasks_per_node=TASKS_PER_NODE,
        cpus_per_task=CPUS_PER_TASKS,
        # gpus_per_node=NGPUS,
        slurm_gres="gpu:2"
        timeout_min=TIMEOUT,
    )

    jobs = []

    ## -- Start a job with <args.num_tasks> task. Each task will process part of the clusters
    with executor.batch():
        for job_start_cluster in range(0, args.num_clusters, args.clusters_per_job):
            exp = SemDeDupJob(args, job_start_cluster)
            job = executor.submit(exp)
            jobs.append(job)
    
    for job in jobs:
        print(f"Submit job id: {job.job_id} for finding duplicates")
        result = job.result()
        
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the config file")
    parser.add_argument("--partition", type=str, default="scaling_data_pruning", help="SLURM partition")
    parser.add_argument("--timeout", type=int, default=60, help="Job timeout")
    parser.add_argument("--cpus-per-task", type=int, default=10, help="Number of CPUs per task")
    parser.add_argument("--ngpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--num-tasks", type=int, default=1, help="Number of tasks")
    args = parser.parse_args()
    launch(args)