import os
import tyro
import sys
import torch
from tqdm import tqdm
import subprocess


file_dir = os.path.dirname(os.path.abspath(__file__))
EVAL_FILE = f"{os.path.join(file_dir, 'run_trained_agent_and_save.py')}"

def find_device_with_least_processes(process_on_gpu):
    num_processes = [len(processes) for processes in process_on_gpu]
    return num_processes.index(min(num_processes))

def check_finished(process_on_gpu):
    for i, processes in enumerate(process_on_gpu):
        for process in processes:
            if process.poll() is not None:
                processes.remove(process)
        
    return process_on_gpu

# run ls in the current directory
def main(path:str, n_parallel:int, horizon:int, n_rollouts:int):
    num_devices = torch.cuda.device_count()
    process_on_gpu = [[] for _ in range(num_devices)]
    # walk through the directory
    for root, dirs, files in tqdm(os.walk(path)):
        for file in files:
            if file.endswith(".pth"):
                while sum([len(processes) for processes in process_on_gpu]) >= n_parallel:
                    process_on_gpu = check_finished(process_on_gpu)
                    
                file_path = os.path.join(root, file)
                print(f"Running rollouts for {file_path}")
                
                cmd = f"python {EVAL_FILE} --agent {file_path} --horizon {horizon} --n_rollouts {n_rollouts}"
                # find the device with the least number of processes
                device_id = find_device_with_least_processes(process_on_gpu)
                cmd = f"CUDA_VISIBLE_DEVICES={device_id} {cmd}"
                # do not show the output of the subprocess
                process = subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
                process_on_gpu[device_id].append(process)
    
    while sum([len(processes) for processes in process_on_gpu]) > 0:
        process_on_gpu = check_finished(process_on_gpu)
        
    print("All rollouts finished!")
                
    
if __name__ == "__main__":
    tyro.cli(main)
