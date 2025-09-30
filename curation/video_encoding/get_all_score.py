import torch.multiprocessing as mp
import curation.video_encoding.suboptimal as suboptimal
import torch
import os

import time
import tyro


dataset_names = [
    "taco_play",
    "jaco_play",
    "berkeley_cable_routing",
    "roboturk",
    "nyu_door_opening_surprising_effectiveness",
    "viola",
    "berkeley_autolab_ur5",
    "toto",
    "language_table",
    "stanford_hydra_dataset_converted_externally_to_rlds",
    "austin_buds_dataset_converted_externally_to_rlds",
    "nyu_franka_play_dataset_converted_externally_to_rlds",
    "furniture_bench_dataset_converted_externally_to_rlds",
    "ucsd_kitchen_dataset_converted_externally_to_rlds",
    "austin_sailor_dataset_converted_externally_to_rlds",
    "austin_sirius_dataset_converted_externally_to_rlds",
    "bc_z",
    "dlr_edan_shared_control_converted_externally_to_rlds",
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds",
    "utaustin_mutex",
    "berkeley_fanuc_manipulation",
    "cmu_stretch",
    "fractal20220817_data",
    "bridge_dataset",
    "kuka",
]

def main(data_dir:str, model_path:str, output_dir:str, batch_size:int=128, viz_num:int=0, viz_thres:float=0.7, split:str="train[:95%]"):
    torch_devices = torch.cuda.device_count()
    print(f"Using {torch_devices} devices")
    job_queue = []
    available_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    for dataset_name in dataset_names:
        while len(available_devices) == 0:
            for job, device in job_queue:
                if not job.is_alive():
                    job_queue.remove((job, device))
                    available_devices.append(device)
            time.sleep(1)
            
        print("-"*50)
        print(f"Starting job for {dataset_name} using device {available_devices[0]}")
        
        torch_device = available_devices.pop(0)
        
        job = mp.Process(target=suboptimal.main, kwargs=dict(
            name = dataset_name,
            data_dir = data_dir,
            model_path = model_path,
            output_dir = output_dir,
            batch_size = batch_size,
            device_id = torch_device,
            split = split,
            visualize = viz_num,
            visualize_thres = viz_thres
        ))
        job.start()
        job_queue.append((job, torch_device))
        
    for job, device in job_queue:
        job.join()
    
if __name__ == "__main__":
    mp.set_start_method("spawn")
    tyro.cli(main)
        
        