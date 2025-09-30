"""
This file contains Dataset classes that are used by torch dataloaders
to fetch batches from hdf5 files.
"""
import os
import time
import h5py
import pickle
import numpy as np
from copy import deepcopy
from contextlib import contextmanager

import torch.utils.data

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.log_utils as LogUtils


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        hdf5_path,
        obs_keys,
        dataset_keys,
        frame_stack=1,
        seq_length=1,
        pad_frame_stack=True,
        pad_seq_length=True,
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=None,
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        filter_by_attribute=None,
        load_next_obs=True,
        curation_args={},
    ):
        """
        Dataset class for fetching sequences of experience.
        Length of the fetched sequence is equal to (@frame_stack - 1 + @seq_length)

        Args:
            hdf5_path (str): path to hdf5

            obs_keys (tuple, list): keys to observation items (image, object, etc) to be fetched from the dataset

            dataset_keys (tuple, list): keys to dataset items (actions, rewards, etc) to be fetched from the dataset

            frame_stack (int): numbers of stacked frames to fetch. Defaults to 1 (single frame).

            seq_length (int): length of sequences to sample. Defaults to 1 (single frame).

            pad_frame_stack (int): whether to pad sequence for frame stacking at the beginning of a demo. This
                ensures that partial frame stacks are observed, such as (s_0, s_0, s_0, s_1). Otherwise, the
                first frame stacked observation would be (s_0, s_1, s_2, s_3).

            pad_seq_length (int): whether to pad sequence for sequence fetching at the end of a demo. This
                ensures that partial sequences at the end of a demonstration are observed, such as
                (s_{T-1}, s_{T}, s_{T}, s_{T}). Otherwise, the last sequence provided would be
                (s_{T-3}, s_{T-2}, s_{T-1}, s_{T}).

            get_pad_mask (bool): if True, also provide padding masks as part of the batch. This can be
                useful for masking loss functions on padded parts of the data.

            goal_mode (str): either "last" or None. Defaults to None, which is to not fetch goals

            hdf5_cache_mode (str): one of ["all", "low_dim", or None]. Set to "all" to cache entire hdf5 
                in memory - this is by far the fastest for data loading. Set to "low_dim" to cache all 
                non-image data. Set to None to use no caching - in this case, every batch sample is 
                retrieved via file i/o. You should almost never set this to None, even for large 
                image datasets.

            hdf5_use_swmr (bool): whether to use swmr feature when opening the hdf5 file. This ensures
                that multiple Dataset instances can all access the same hdf5 file without problems.

            hdf5_normalize_obs (bool): if True, normalize observations by computing the mean observation
                and std of each observation (in each dimension and modality), and normalizing to unit
                mean and variance in each dimension.

            filter_by_attribute (str): if provided, use the provided filter key to look up a subset of
                demonstrations to load

            load_next_obs (bool): whether to load next_obs from the dataset
        """
        super(SequenceDataset, self).__init__()

        self.hdf5_path = os.path.expanduser(hdf5_path)
        self.curation_args = curation_args
        self.hdf5_use_swmr = hdf5_use_swmr
        self.hdf5_normalize_obs = hdf5_normalize_obs
        self._hdf5_file = None

        assert hdf5_cache_mode in ["all", "low_dim", None]
        self.hdf5_cache_mode = hdf5_cache_mode

        self.load_next_obs = load_next_obs
        self.filter_by_attribute = filter_by_attribute

        # get all keys that needs to be fetched
        self.obs_keys = tuple(obs_keys)
        self.dataset_keys = tuple(dataset_keys)

        self.n_frame_stack = frame_stack
        assert self.n_frame_stack >= 1

        self.seq_length = seq_length
        assert self.seq_length >= 1

        self.goal_mode = goal_mode
        if self.goal_mode is not None:
            assert self.goal_mode in ["last"]
        if not self.load_next_obs:
            assert self.goal_mode != "last"  # we use last next_obs as goal

        self.pad_seq_length = pad_seq_length
        self.pad_frame_stack = pad_frame_stack
        self.get_pad_mask = get_pad_mask

        self.load_demo_info(filter_by_attribute=self.filter_by_attribute)

        # maybe prepare for observation normalization
        self.obs_normalization_stats = None
        if self.hdf5_normalize_obs:
            self.obs_normalization_stats = self.normalize_obs()

        # maybe store dataset in memory for fast access
        if self.hdf5_cache_mode in ["all", "low_dim"]:
            obs_keys_in_memory = self.obs_keys
            if self.hdf5_cache_mode == "low_dim":
                # only store low-dim observations
                obs_keys_in_memory = []
                for k in self.obs_keys:
                    if ObsUtils.key_is_obs_modality(k, "low_dim"):
                        obs_keys_in_memory.append(k)
            self.obs_keys_in_memory = obs_keys_in_memory

            self.hdf5_cache = self.load_dataset_in_memory(
                demo_list=self.demos,
                hdf5_file=self.hdf5_file,
                obs_keys=self.obs_keys_in_memory,
                dataset_keys=self.dataset_keys,
                load_next_obs=self.load_next_obs
            )

            if self.hdf5_cache_mode == "all":
                # cache getitem calls for even more speedup. We don't do this for
                # "low-dim" since image observations require calls to getitem anyways.
                print("SequenceDataset: caching get_item calls...")
                self.getitem_cache = [self.get_item(i) for i in LogUtils.custom_tqdm(range(len(self)))]

                # don't need the previous cache anymore
                del self.hdf5_cache
                self.hdf5_cache = None
        else:
            self.hdf5_cache = None

        self.close_and_delete_hdf5_handle()
    
    def subop_score_minus_bias(self, score, subop_arg, bias_array):
        for j in range(len(subop_arg['rank_thres'])-1):
            start, end = subop_arg['rank_thres'][j], subop_arg['rank_thres'][j+1]
            start_idx, end_idx = int(start*20), int(end*20)
            score[-end_idx:-start_idx] -= bias_array[j]
        return score
    
    def compute_subop_thres_and_stepbias(self, hdf5_file, demos, subop_arg):
        if subop_arg.get("percentile_as_threshold", False):
            print("####Leo's log####: Using percentile as threshold")
            return subop_arg.get("subop_percentile"), None

        assert subop_arg.get("subop_percentile") is not None, "subop_percentile is not provided, please provide it in curation_args"
        percentile = subop_arg.get("subop_percentile")
        scores = []
        for ep in demos:
            demo = hdf5_file["data/{}".format(ep)]
            score = np.squeeze(demo['subop_score'][:])
            scores.append(score)
        
        # compute the bias for each rank
        mean_array = None
        if subop_arg.get("minus_step_bias", False):
            print("Computing step bias...")
            mean_array = np.zeros(len(subop_arg['rank_thres'])-1)
            rank_cnt = np.zeros(len(subop_arg['rank_thres'])-1)
            
            for i, score in enumerate(scores):
                for j in range(len(subop_arg['rank_thres'])-1):
                    start, end = subop_arg['rank_thres'][j], subop_arg['rank_thres'][j+1]
                    start_idx, end_idx = int(start*20), int(end*20)
                    mean_array[j] += np.sum(score[-end_idx:-start_idx])
                    rank_cnt[j] += end_idx - start_idx
            mean_array /= rank_cnt
        
        print("Computing subop threshold...")
        for i, score in enumerate(scores):
            if subop_arg.get("minus_step_bias", False):
                score = self.subop_score_minus_bias(score, subop_arg, mean_array)
            scores[i] = self.smooth_score(score, subop_arg.get("smooth_window", 1), subop_arg.get("traj_level", False), subop_arg.get("mix_level", 0.0))
        scores = np.concatenate(scores)
        
        thres = np.percentile(scores, percentile * 100)
        return thres, mean_array
    
    def smooth_score(self, score, window, traj_level=False, mix_level=0.0):
        if traj_level:
            smoothed_score = np.ones_like(score) * np.mean(score)
            return smoothed_score
        if window >= 1:
            window = min(window, len(score))
            cum_cnt = np.convolve(np.ones_like(score), np.ones(window), mode='same')
            cum_sum = np.convolve(score, np.ones(window), mode='same')
            smoothed_score = cum_sum / cum_cnt
        else:
            smoothed_score = score
            
        if mix_level > 0.0:
            smoothed_score = np.mean(score) * mix_level + smoothed_score * (1 - mix_level)
        return smoothed_score
    
    def count_average_length(self, hdf5_file, demos):
        total_length = 0
        for ep in demos:
            total_length += hdf5_file["data/{}".format(ep)].attrs["num_samples"]
        return total_length//len(demos)
    
    def compute_dedup_mask(self, hdf5_file, demos, dedup_arg):
        dedup_type = dedup_arg.get("dedup_type", "density")
        print("Dedup type: ", dedup_type)
        assert dedup_type in ["density", "semantic", "threshold", "mutal_info"], "dedup_type should be one of ['density', 'semantic', 'threshold']"
        random_dedup = dedup_arg.get("random_dedup", False)
        if dedup_type == "density":
            print("Density deduplication")
            densitys = []
            idx_to_ep = {}
            d_idx = 0
            for ep in demos:
                density = np.squeeze(hdf5_file[f"data/{ep}/density"][:])
                density_idx = np.squeeze(hdf5_file[f"data/{ep}/density_idx"][:])
                if len(density_idx.shape) > 1:
                    for start, end in density_idx:
                        idx_to_ep[d_idx] = (ep, start, end)
                        d_idx += 1
                else:
                    density = np.array([density])
                    start, end = density_idx
                    idx_to_ep[d_idx] = (ep, start, end)
                    d_idx += 1
                densitys.append(density)
            densitys = np.concatenate(densitys)
            assert len(densitys) == len(idx_to_ep)
            
            num_keep_samples = int(dedup_arg['keep_ratio'] * len(densitys))
            inverse_densitys = 1/densitys
            prob = inverse_densitys/np.sum(inverse_densitys)
            np_rng = np.random.RandomState(0)
            
            if random_dedup:
                print("Random deduplication")
                prob = np.ones(len(densitys))/len(densitys)
            chosen_idx = np_rng.choice(len(densitys), num_keep_samples, replace=False, p=prob) # choose keep idx based on inverse density

            # concat_dedup_mask = np.ones(len(densitys), dtype=bool)
            # concat_dedup_mask[chosen_idx] = False
        elif dedup_type == "semantic":
            print("Semantic deduplication")
            max_sim = []
            idx_to_ep = {}
            s_idx = 0
            for ep in demos:
                sim = np.squeeze(hdf5_file[f"data/{ep}/max_sim"][:])
                sim_idx = np.squeeze(hdf5_file[f"data/{ep}/max_sim_idx"][:])
                if len(sim_idx.shape) > 1:
                    for start, end in sim_idx:
                        idx_to_ep[s_idx] = (ep, start, end)
                        s_idx += 1
                else:
                    sim = np.array([sim])
                    start, end = sim_idx
                    idx_to_ep[s_idx] = (ep, start, end)
                    s_idx += 1
                max_sim.append(sim)
            max_sim = np.concatenate(max_sim)
            assert len(max_sim) == len(idx_to_ep)

            # sort max_sim's index using max_sim value
            idx = np.argsort(max_sim)
            if random_dedup:
                print("Random deduplication")
                np_rng = np.random.RandomState(0)
                np_rng.shuffle(idx)
            num_keep_samples = int(dedup_arg['keep_ratio'] * len(max_sim))
            print("######### Dedup Threshold: ", max_sim[idx[num_keep_samples]])
            chosen_idx = idx[:num_keep_samples]

        elif dedup_type == "threshold":
            print("####Leo's log####: Semantic deduplication using threshold")
            max_sim = []
            idx_to_ep = {}
            s_idx = 0
            for ep in demos:
                sim = np.squeeze(hdf5_file[f"data/{ep}/max_sim"][:])
                sim_idx = np.squeeze(hdf5_file[f"data/{ep}/max_sim_idx"][:])
                if len(sim_idx.shape) > 1:
                    for start, end in sim_idx:
                        idx_to_ep[s_idx] = (ep, start, end)
                        s_idx += 1
                else:
                    sim = np.array([sim])
                    start, end = sim_idx
                    idx_to_ep[s_idx] = (ep, start, end)
                    s_idx += 1
                max_sim.append(sim)
            max_sim = np.concatenate(max_sim)
            assert len(max_sim) == len(idx_to_ep)

            # sort max_sim's index using max_sim value
            idx = np.argsort(max_sim)
            if random_dedup:
                print("Random deduplication")
                np_rng = np.random.RandomState(0)
                np_rng.shuffle(idx)
            # num_keep_samples = int(dedup_arg['keep_ratio'] * len(max_sim))
            # print("######### Dedup Threshold: ", max_sim[idx[num_keep_samples]])
            # chosen_idx = idx[:num_keep_samples]
            # the keep_ratio is now the threshold
            threshold = dedup_arg['keep_ratio']
            print("######### Dedup Threshold: ", threshold)
            chosen_idx = np.argwhere(max_sim < threshold).flatten()
            print(f'###### semantic curated samples: {np.sum(max_sim < threshold) / len(max_sim)}')
        elif dedup_type == "mutal_info":
            seed = dedup_arg.get("seed", 1)
            scores = []
            idx_to_ep = {}
            m_idx = 0
            for ep in demos:
                score = hdf5_file[f"data/{ep}/mutal_info_seed_{seed}"][()]
                scores.append(score)
                idx_to_ep[m_idx] = (ep, 0, len(hdf5_file[f"data/{ep}/actions"]))
                m_idx += 1
            scores = np.array(scores)
            assert len(scores) == len(idx_to_ep)
            # sort max_sim's index using max_sim value
            idx = np.argsort(scores)[::-1]
            num_keep_samples = int(dedup_arg['keep_ratio'] * len(scores))
            print("######### Dedup Threshold: ", scores[idx[num_keep_samples]])
            chosen_idx = idx[:num_keep_samples]
            
        dedup_mask = {}
        # initialize for all the episodes
        for ep in demos:
            demo_length = hdf5_file["data/{}".format(ep)].attrs["num_samples"]
            dedup_mask[ep] = np.ones(demo_length, dtype=bool)  
        # mask out all selected idx
        for idx in chosen_idx:
            ep, start, end = idx_to_ep[idx]
            dedup_mask[ep][start:end] = False
        return dedup_mask
    
    def load_demo_info(self, filter_by_attribute=None, demos=None):
        """
        Args:
            filter_by_attribute (str): if provided, use the provided filter key
                to select a subset of demonstration trajectories to load

            demos (list): list of demonstration keys to load from the hdf5 file. If 
                omitted, all demos in the file (or under the @filter_by_attribute 
                filter key) are used.
        """
        # filter demo trajectory by mask
        if demos is not None:
            self.demos = demos
        elif filter_by_attribute is not None:
            self.demos = [elem.decode("utf-8") for elem in np.array(self.hdf5_file["mask/{}".format(filter_by_attribute)][:])]
        else:
            self.demos = list(self.hdf5_file["data"].keys())

        # sort demo keys
        inds = np.argsort([int(elem[5:]) for elem in self.demos])
        self.demos = [self.demos[i] for i in inds]

        self.n_demos = len(self.demos)

        # keep internal index maps to know which transitions belong to which demos
        self._index_to_demo_id = dict()  # maps every index to a demo id
        self._demo_id_to_start_indices = dict()  # gives start index per demo id
        self._demo_id_to_demo_length = dict()
        self.curation_mask = {}
        self._demo_id_to_demo_length_after_curate = dict()
        
        # compute subop thres according to percentile
        if self.curation_args.get("subop_curate", {}).get("enabled", False):
            subop_thres, step_bias = self.compute_subop_thres_and_stepbias(self.hdf5_file, self.demos, self.curation_args.get("subop_curate", {}))
            print("######### Subop Threshold: ", subop_thres)
            average_length = self.count_average_length(self.hdf5_file, self.demos)
            self.kept_bin_count = np.zeros(average_length//10, dtype=int)
            
        if self.curation_args.get("dedup_curate", {}).get("enabled", False):
            dedup_mask = self.compute_dedup_mask(self.hdf5_file, self.demos, self.curation_args.get("dedup_curate", {}))

        # determine index mapping
        self.total_num_sequences = 0
        for ep in self.demos:
            demo_length = self.hdf5_file["data/{}".format(ep)].attrs["num_samples"]
            self._demo_id_to_start_indices[ep] = self.total_num_sequences
            self._demo_id_to_demo_length[ep] = demo_length
            self.curation_mask[ep] = np.zeros(demo_length, dtype=bool)
            
            if self.curation_args.get("subop_curate", {}).get("enabled", False):
                subop_score = np.squeeze(self.hdf5_file["data/{}".format(ep)]["subop_score"][:])
                if self.curation_args.get("subop_curate", {}).get("minus_step_bias", False):
                    subop_score = self.subop_score_minus_bias(subop_score, self.curation_args['subop_curate'], step_bias)
                subop_score = self.smooth_score(subop_score, self.curation_args['subop_curate'].get("smooth_window", 1), self.curation_args['subop_curate'].get("traj_level", False), self.curation_args['subop_curate'].get("mix_level", 0.0))
                self.curation_mask[ep] = (subop_score > subop_thres) | self.curation_mask[ep]
                kept_norm_idx = np.argwhere(~self.curation_mask[ep]).flatten()/demo_length
                bin_count = np.histogram(kept_norm_idx, bins=np.arange(0, 1.0, 1/(len(self.kept_bin_count)+1)))[0]
                self.kept_bin_count += bin_count
            
            if self.curation_args.get("dedup_curate", {}).get("enabled", False):
                self.curation_mask[ep] = self.curation_mask[ep] | dedup_mask[ep]
                
                # idx = int(ep.split("_")[-1])
                # if idx >= 200: self.curation_mask[ep] = np.ones(demo_length, dtype=bool)
                
            demo_length_after_curation = np.sum(~self.curation_mask[ep])
            
            self._demo_id_to_demo_length_after_curate[ep] = demo_length_after_curation
            demo_length = demo_length_after_curation
            
            if np.all(self.curation_mask[ep]):
                continue # skip the demo if all the transitions are removed    
                

            num_sequences = demo_length
            # determine actual number of sequences taking into account whether to pad for frame_stack and seq_length
            if not self.pad_frame_stack:
                num_sequences -= (self.n_frame_stack - 1)
            if not self.pad_seq_length:
                num_sequences -= (self.seq_length - 1)

            if self.pad_seq_length:
                assert demo_length >= 1  # sequence needs to have at least one sample
                num_sequences = max(num_sequences, 1)
            else:
                assert num_sequences >= 1  # assume demo_length >= (self.n_frame_stack - 1 + self.seq_length)

            for _ in range(num_sequences):
                self._index_to_demo_id[self.total_num_sequences] = ep
                self.total_num_sequences += 1
                
        if self.curation_args.get("subop_curate", {}).get("enabled", False) or self.curation_args.get("dedup_curate", {}).get("enabled", False):
            total_length, total_curated_length = 0, 0
            for ep in self.demos:
                total_length += self._demo_id_to_demo_length[ep]
                total_curated_length += self._demo_id_to_demo_length_after_curate.get(ep, self._demo_id_to_demo_length[ep])
            print(f"{self.hdf5_path}\nTotal number of samples: {total_length}, Total number of samples after curated: {total_curated_length}. Ratio: {total_curated_length/total_length:.4f}")
    @property
    def hdf5_file(self):
        """
        This property allows for a lazy hdf5 file open.
        """
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=self.hdf5_use_swmr, libver='latest')
        return self._hdf5_file

    def close_and_delete_hdf5_handle(self):
        """
        Maybe close the file handle.
        """
        if self._hdf5_file is not None:
            self._hdf5_file.close()
        self._hdf5_file = None

    @contextmanager
    def hdf5_file_opened(self):
        """
        Convenient context manager to open the file on entering the scope
        and then close it on leaving.
        """
        should_close = self._hdf5_file is None
        yield self.hdf5_file
        if should_close:
            self.close_and_delete_hdf5_handle()

    def __del__(self):
        self.close_and_delete_hdf5_handle()

    def __repr__(self):
        """
        Pretty print the class and important attributes on a call to `print`.
        """
        msg = str(self.__class__.__name__)
        msg += " (\n\tpath={}\n\tobs_keys={}\n\tseq_length={}\n\tfilter_key={}\n\tframe_stack={}\n"
        msg += "\tpad_seq_length={}\n\tpad_frame_stack={}\n\tgoal_mode={}\n"
        msg += "\tcache_mode={}\n"
        msg += "\tnum_demos={}\n\tnum_sequences={}\n)"
        filter_key_str = self.filter_by_attribute if self.filter_by_attribute is not None else "none"
        goal_mode_str = self.goal_mode if self.goal_mode is not None else "none"
        cache_mode_str = self.hdf5_cache_mode if self.hdf5_cache_mode is not None else "none"
        msg = msg.format(self.hdf5_path, self.obs_keys, self.seq_length, filter_key_str, self.n_frame_stack,
                         self.pad_seq_length, self.pad_frame_stack, goal_mode_str, cache_mode_str,
                         self.n_demos, self.total_num_sequences)
        return msg

    def __len__(self):
        """
        Ensure that the torch dataloader will do a complete pass through all sequences in 
        the dataset before starting a new iteration.
        """
        return self.total_num_sequences

    def load_dataset_in_memory(self, demo_list, hdf5_file, obs_keys, dataset_keys, load_next_obs):
        """
        Loads the hdf5 dataset into memory, preserving the structure of the file. Note that this
        differs from `self.getitem_cache`, which, if active, actually caches the outputs of the
        `getitem` operation.

        Args:
            demo_list (list): list of demo keys, e.g., 'demo_0'
            hdf5_file (h5py.File): file handle to the hdf5 dataset.
            obs_keys (list, tuple): observation keys to fetch, e.g., 'images'
            dataset_keys (list, tuple): dataset keys to fetch, e.g., 'actions'
            load_next_obs (bool): whether to load next_obs from the dataset

        Returns:
            all_data (dict): dictionary of loaded data.
        """
        all_data = dict()
        print("SequenceDataset: loading dataset into memory...")
        for ep in LogUtils.custom_tqdm(demo_list):
            all_data[ep] = {}
            all_data[ep]["attrs"] = {}
            all_data[ep]["attrs"]["num_samples"] = hdf5_file["data/{}".format(ep)].attrs["num_samples"]
            # get obs
            all_data[ep]["obs"] = {k: hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype('float32') for k in obs_keys}
            if load_next_obs:
                all_data[ep]["next_obs"] = {k: hdf5_file["data/{}/next_obs/{}".format(ep, k)][()].astype('float32') for k in obs_keys}
            # get other dataset keys
            for k in dataset_keys:
                if k in hdf5_file["data/{}".format(ep)]:
                    all_data[ep][k] = hdf5_file["data/{}/{}".format(ep, k)][()].astype('float32')
                else:
                    all_data[ep][k] = np.zeros((all_data[ep]["attrs"]["num_samples"], 1), dtype=np.float32)

            if "model_file" in hdf5_file["data/{}".format(ep)].attrs:
                all_data[ep]["attrs"]["model_file"] = hdf5_file["data/{}".format(ep)].attrs["model_file"]

        return all_data

    def normalize_obs(self):
        """
        Computes a dataset-wide mean and standard deviation for the observations 
        (per dimension and per obs key) and returns it.
        """
        def _compute_traj_stats(traj_obs_dict):
            """
            Helper function to compute statistics over a single trajectory of observations.
            """
            traj_stats = { k : {} for k in traj_obs_dict }
            for k in traj_obs_dict:
                traj_stats[k]["n"] = traj_obs_dict[k].shape[0]
                traj_stats[k]["mean"] = traj_obs_dict[k].mean(axis=0, keepdims=True) # [1, ...]
                traj_stats[k]["sqdiff"] = ((traj_obs_dict[k] - traj_stats[k]["mean"]) ** 2).sum(axis=0, keepdims=True) # [1, ...]
            return traj_stats

        def _aggregate_traj_stats(traj_stats_a, traj_stats_b):
            """
            Helper function to aggregate trajectory statistics.
            See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
            for more information.
            """
            merged_stats = {}
            for k in traj_stats_a:
                n_a, avg_a, M2_a = traj_stats_a[k]["n"], traj_stats_a[k]["mean"], traj_stats_a[k]["sqdiff"]
                n_b, avg_b, M2_b = traj_stats_b[k]["n"], traj_stats_b[k]["mean"], traj_stats_b[k]["sqdiff"]
                n = n_a + n_b
                mean = (n_a * avg_a + n_b * avg_b) / n
                delta = (avg_b - avg_a)
                M2 = M2_a + M2_b + (delta ** 2) * (n_a * n_b) / n
                merged_stats[k] = dict(n=n, mean=mean, sqdiff=M2)
            return merged_stats

        # Run through all trajectories. For each one, compute minimal observation statistics, and then aggregate
        # with the previous statistics.
        ep = self.demos[0]
        obs_traj = {k: self.hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype('float32') for k in self.obs_keys}
        obs_traj = ObsUtils.process_obs_dict(obs_traj)
        merged_stats = _compute_traj_stats(obs_traj)
        print("SequenceDataset: normalizing observations...")
        for ep in LogUtils.custom_tqdm(self.demos[1:]):
            obs_traj = {k: self.hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype('float32') for k in self.obs_keys}
            obs_traj = ObsUtils.process_obs_dict(obs_traj)
            traj_stats = _compute_traj_stats(obs_traj)
            merged_stats = _aggregate_traj_stats(merged_stats, traj_stats)

        obs_normalization_stats = { k : {} for k in merged_stats }
        for k in merged_stats:
            # note we add a small tolerance of 1e-3 for std
            obs_normalization_stats[k]["mean"] = merged_stats[k]["mean"]
            obs_normalization_stats[k]["std"] = np.sqrt(merged_stats[k]["sqdiff"] / merged_stats[k]["n"]) + 1e-3
        return obs_normalization_stats

    def get_obs_normalization_stats(self):
        """
        Returns dictionary of mean and std for each observation key if using
        observation normalization, otherwise None.

        Returns:
            obs_normalization_stats (dict): a dictionary for observation
                normalization. This maps observation keys to dicts
                with a "mean" and "std" of shape (1, ...) where ... is the default
                shape for the observation.
        """
        assert self.hdf5_normalize_obs, "not using observation normalization!"
        return deepcopy(self.obs_normalization_stats)

    def get_dataset_for_ep(self, ep, key):
        """
        Helper utility to get a dataset for a specific demonstration.
        Takes into account whether the dataset has been loaded into memory.
        """

        # check if this key should be in memory
        key_should_be_in_memory = (self.hdf5_cache_mode in ["all", "low_dim"])
        if key_should_be_in_memory:
            # if key is an observation, it may not be in memory
            if '/' in key:
                key1, key2 = key.split('/')
                assert(key1 in ['obs', 'next_obs'])
                if key2 not in self.obs_keys_in_memory:
                    key_should_be_in_memory = False

        if key_should_be_in_memory:
            # read cache
            if '/' in key:
                key1, key2 = key.split('/')
                assert(key1 in ['obs', 'next_obs'])
                ret = self.hdf5_cache[ep][key1][key2]
            else:
                ret = self.hdf5_cache[ep][key]
        else:
            # read from file
            hd5key = "data/{}/{}".format(ep, key)
            ret = self.hdf5_file[hd5key]
        return ret

    def __getitem__(self, index):
        """
        Fetch dataset sequence @index (inferred through internal index map), using the getitem_cache if available.
        """
        if self.hdf5_cache_mode == "all":
            return self.getitem_cache[index]
        return self.get_item(index)

    def get_item(self, index):
        """
        Main implementation of getitem when not using cache.
        """

        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        if self.curation_args.get("subop_curate", {}).get("enabled", False) == False and self.curation_args.get("dedup_curate", {}).get("enabled", False) == False:
            index_in_demo = index - demo_start_index + demo_index_offset
        else: # if curation is enabled
            curated_demo_length = self._demo_id_to_demo_length_after_curate[demo_id]
            demo_length = curated_demo_length
            assert demo_length > 0 and curated_demo_length > 0, "demo length is 0, check the deduplication process in load demo info"
            not_curated_indices = np.argwhere(~self.curation_mask[demo_id]).flatten()
            index_in_demo = not_curated_indices[index - demo_start_index] + demo_index_offset

        # end at offset index if not padding for seq length
        demo_length_offset = 0 if self.pad_seq_length else (self.seq_length - 1)
        end_index_in_demo = demo_length - demo_length_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.dataset_keys,
            seq_length=self.seq_length
        )

        # determine goal index
        goal_index = None
        if self.goal_mode == "last":
            goal_index = end_index_in_demo - 1

        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            seq_length=self.seq_length,
            prefix="obs"
        )
        if self.hdf5_normalize_obs:
            meta["obs"] = ObsUtils.normalize_obs(meta["obs"], obs_normalization_stats=self.obs_normalization_stats)

        if self.load_next_obs:
            meta["next_obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=index_in_demo,
                keys=self.obs_keys,
                num_frames_to_stack=self.n_frame_stack - 1,
                seq_length=self.seq_length,
                prefix="next_obs"
            )
            if self.hdf5_normalize_obs:
                meta["next_obs"] = ObsUtils.normalize_obs(meta["next_obs"], obs_normalization_stats=self.obs_normalization_stats)

        if goal_index is not None:
            goal = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=goal_index,
                keys=self.obs_keys,
                num_frames_to_stack=0,
                seq_length=1,
                prefix="next_obs",
            )
            if self.hdf5_normalize_obs:
                goal = ObsUtils.normalize_obs(goal, obs_normalization_stats=self.obs_normalization_stats)
            meta["goal_obs"] = {k: goal[k][0] for k in goal}  # remove sequence dimension for goal

        return meta

    def get_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1):
        """
        Extract a (sub)sequence of data items from a demo given the @keys of the items.

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range

        Returns:
            a dictionary of extracted items.
        """
        assert num_frames_to_stack >= 0
        assert seq_length >= 1

        demo_length = self._demo_id_to_demo_length[demo_id]
        assert index_in_demo < demo_length

        # determine begin and end of sequence
        seq_begin_index = max(0, index_in_demo - num_frames_to_stack)
        seq_end_index = min(demo_length, index_in_demo + seq_length)

        # determine sequence padding
        seq_begin_pad = max(0, num_frames_to_stack - index_in_demo)  # pad for frame stacking
        seq_end_pad = max(0, index_in_demo + seq_length - demo_length)  # pad for sequence length

        # make sure we are not padding if specified.
        if not self.pad_frame_stack:
            assert seq_begin_pad == 0
        if not self.pad_seq_length:
            assert seq_end_pad == 0

        # fetch observation from the dataset file
        seq = dict()
        for k in keys:
            data = self.get_dataset_for_ep(demo_id, k)
            seq[k] = data[seq_begin_index: seq_end_index].astype("float32")

        seq = TensorUtils.pad_sequence(seq, padding=(seq_begin_pad, seq_end_pad), pad_same=True)
        pad_mask = np.array([0] * seq_begin_pad + [1] * (seq_end_index - seq_begin_index) + [0] * seq_end_pad)
        pad_mask = pad_mask[:, None].astype(bool)

        return seq, pad_mask

    def get_obs_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1, prefix="obs"):
        """
        Extract a (sub)sequence of observation items from a demo given the @keys of the items.

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range
            prefix (str): one of "obs", "next_obs"

        Returns:
            a dictionary of extracted items.
        """
        obs, pad_mask = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=tuple('{}/{}'.format(prefix, k) for k in keys),
            num_frames_to_stack=num_frames_to_stack,
            seq_length=seq_length,
        )
        obs = {k.split('/')[1]: obs[k] for k in obs}  # strip the prefix
        if self.get_pad_mask:
            obs["pad_mask"] = pad_mask

        # prepare image observations from dataset
        return ObsUtils.process_obs_dict(obs)

    def get_dataset_sequence_from_demo(self, demo_id, index_in_demo, keys, seq_length=1):
        """
        Extract a (sub)sequence of dataset items from a demo given the @keys of the items (e.g., states, actions).
        
        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range

        Returns:
            a dictionary of extracted items.
        """
        data, pad_mask = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=keys,
            num_frames_to_stack=0,  # don't frame stack for meta keys
            seq_length=seq_length,
        )
        if self.get_pad_mask:
            data["pad_mask"] = pad_mask
        return data

    def get_trajectory_at_index(self, index):
        """
        Method provided as a utility to get an entire trajectory, given
        the corresponding @index.
        """
        demo_id = self.demos[index]
        demo_length = self._demo_id_to_demo_length[demo_id]

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=0,
            keys=self.dataset_keys,
            seq_length=demo_length
        )
        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=0,
            keys=self.obs_keys,
            seq_length=demo_length
        )
        if self.load_next_obs:
            meta["next_obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=0,
                keys=self.obs_keys,
                seq_length=demo_length,
                prefix="next_obs"
            )

        meta["ep"] = demo_id
        return meta

    def get_dataset_sampler(self):
        """
        Return instance of torch.utils.data.Sampler or None. Allows
        for dataset to define custom sampling logic, such as
        re-weighting the probability of samples being drawn.
        See the `train` function in scripts/train.py, and torch
        `DataLoader` documentation, for more info.
        """
        if self.curation_args.get("subop_curate", {}).get("enabled", False) == False or self.curation_args.get("subop_curate", {}).get("rebalance", False) == False:
            return None
        else:
            data_density = self.kept_bin_count
            inverse_density = 1 / data_density
            weight = inverse_density / np.sum(inverse_density)
            weight_array = np.zeros(self.total_num_sequences)
            demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
            for index in range(len(weight_array)):
                demo_id = self._index_to_demo_id[index]
                demo_start_index = self._demo_id_to_start_indices[demo_id]
                curated_demo_length = self._demo_id_to_demo_length_after_curate[demo_id]
                demo_length = curated_demo_length
                assert demo_length > 0 and curated_demo_length > 0, "demo length is 0, check the deduplication process in load demo info"
                not_curated_indices = np.argwhere(~self.curation_mask[demo_id]).flatten()
                index_in_demo = not_curated_indices[index - demo_start_index] + demo_index_offset
                weight_array[index] = weight[int(index_in_demo/len(self.curation_mask[demo_id])*len(weight))]
            sampler = torch.utils.data.WeightedRandomSampler(weight_array, len(weight_array), replacement=True)
            return sampler
