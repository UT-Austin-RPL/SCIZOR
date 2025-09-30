import os
from curation.suboptimal_classifier.dataset.dataset import AugActionDataset, ExternIterator 
from curation.suboptimal_classifier.discriminator.discriminator import Discriminator
from curation.suboptimal_classifier.evaluate import get_dataloader_for_eval, evaluate_dataloader, evaluate_dataset
from octo.octo.utils.train_utils import Timer

from absl import flags, app
from ml_collections import config_flags

import torch
from tqdm import tqdm
import wandb
from curation.suboptimal_classifier.utils import wandb_log, wandb_init, TrainInfo, Torch_Preallocater
from accelerate import Accelerator
import torch.multiprocessing as mp
import torch.distributed as dist
import json
from curation.suboptimal_classifier.dataset.dataset import run_zmq_dataset
import zmq
import pickle
from accelerate import DistributedDataParallelKwargs as DDPK
kwargs = DDPK(find_unused_parameters=True)

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")
flags.DEFINE_string("port", "4399", "Port for zmq communication")

config_dir = os.path.join(os.path.dirname(__file__), "config")
config_flags.DEFINE_config_file(
    "config",
    os.path.join(config_dir, "config.py"),
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

timer = Timer()
accelerator = Accelerator(kwargs_handlers=[kwargs])

def create_dataset(config):
    extern_source = ExternIterator(config)
    aug_action_dataset = AugActionDataset(config, extern_source, **config.dali_kwargs)
    # next(aug_action_dataset)
    return aug_action_dataset

def start_zmq_dataset(config, i):
    zmq_process = mp.Process(target=run_zmq_dataset, args=(config, str(int(FLAGS.port)+i)))
    zmq_process.start()
    
def create_zmq_dealer(id, num_dataset):
    context = zmq.Context()
    #consumers
    socket = context.socket(zmq.DEALER)
    socket.connect(f"tcp://localhost:{str(int(FLAGS.port)+id%num_dataset)}")
    socket.setsockopt_string(zmq.IDENTITY, str(id))
    return socket

def receive_data(socket:zmq.Socket):
    socket.send(b'Request for batch')
    empty, pickle_batch = socket.recv_multipart()
    batch = pickle.loads(pickle_batch)
    return batch
    # return None

def main(_):
    torch.manual_seed(FLAGS.config.seed + accelerator.process_index)
        
    discriminator = Discriminator(**FLAGS.config.discriminator)
    
    device = accelerator.device
    device_id = device.index if device.index is not None else 0
    
    preallocater = Torch_Preallocater(0.6, [torch.device(f"cuda:{device_id}")])
    if accelerator.is_main_process:
        wandb_id = wandb_init(FLAGS.config, FLAGS.name, FLAGS.debug)
        
        save_dir = FLAGS.config.save_dir
        if save_dir is not None:
            exp_dir = os.path.join(save_dir, (wandb_id + "_debug") if FLAGS.debug else wandb_id)
            if not os.path.exists(exp_dir):
                os.makedirs(exp_dir)
            # Save config
            with open(os.path.join(exp_dir, "config.json"), "w") as f:
                json.dump(FLAGS.config.to_dict(), f, indent=4)
        else:
            exp_dir = None
        
        for i in range(FLAGS.config.num_datasets):
            start_zmq_dataset(FLAGS.config, i) 
                   
    accelerator.wait_for_everyone()
    process_index = accelerator.device.index if accelerator.device.index is not None else 0
    data_socket = create_zmq_dealer(process_index, FLAGS.config.num_datasets)
    print(f"current device: {accelerator.device} created zmq dealer!!!!!!!!!!!!!!!!!!!!!")
    accelerator.wait_for_everyone()
    preallocater.free()
            
        
    discriminator = Discriminator(**FLAGS.config.discriminator)
    
    if FLAGS.config.load_path is not None:
        discriminator.load_state_dict(torch.load(FLAGS.config.load_path))
        
    optimizer = torch.optim.AdamW(discriminator.parameters(), **FLAGS.config.optimizer)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, FLAGS.config.optimizer.lr, total_steps=FLAGS.config.num_steps*accelerator.num_processes, pct_start=FLAGS.config.scheduler.pct_start)
    
    if FLAGS.config.mixed_precision:
        scaler = torch.GradScaler()
    else:
        scaler = None
    
    num_devices = torch.cuda.device_count()   
    discriminator, optimizer, scheduler = accelerator.prepare(discriminator, optimizer, scheduler)
    discriminator.train()
    
    def train_step(discriminator:Discriminator, batch, optimizer, scheduler, train_info, step, scaler):
        images, actions, action_scores, sub_scores ,task_desc, norm_delta_proprio, delta_grip_act = tuple(batch)
        
        if scaler is not None:
            images, action_scores, actions, norm_delta_proprio, delta_grip_act = map(lambda x: x.half(), [images, action_scores, actions, norm_delta_proprio, delta_grip_act])
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                result = discriminator.forward(images, task_desc, actions, score=action_scores, norm_delta_proprio=norm_delta_proprio, delta_grip_act=delta_grip_act, training=True)
            loss_item = result["loss"].item()
            result["loss"] = scaler.scale(result["loss"])
        else:
            result = discriminator.forward(images, task_desc, actions, score=action_scores, norm_delta_proprio=norm_delta_proprio, delta_grip_act=delta_grip_act, training=True)
            loss_item = result["loss"].item()
        
        accelerator.backward(result["loss"]/FLAGS.config.grad_accum_steps)
        
        if (step+1) % FLAGS.config.grad_accum_steps == 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            
        scheduler.step()
        
        del result["output"]
        train_info.update(result["metrics"])
            
        return loss_item
    
    train_info = TrainInfo()
    
    if accelerator.is_main_process:
        tqdm_bar = tqdm(range(FLAGS.config.num_steps))
    else:
        tqdm_bar = range(FLAGS.config.num_steps)
        
    for i in tqdm_bar:
        with timer("datset"):
            batch = receive_data(data_socket)
            batch_on_device = map(lambda x: x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x, batch)                
        with timer("train"):
            loss = train_step(discriminator, batch_on_device, optimizer, scheduler, train_info, i, scaler)

        if accelerator.is_main_process:
            tqdm_bar.set_postfix({f"loss_{FLAGS.config.discriminator.loss_fn_type}": loss})
            if (i+1) % FLAGS.config.log_interval == 0 and not FLAGS.debug:
                wandb_log({**train_info.get_average(), "lr": optimizer.param_groups[0]["lr"], 'time': timer.get_average_times()}, i+1)
            
        if (i+1) % FLAGS.config.save_interval == 0:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process and exp_dir is not None:
                state_dict = accelerator.unwrap_model(discriminator).state_dict()
                torch.save(state_dict, os.path.join(exp_dir, f"model_{i+1}.pth"))
            
        if (i+1) % FLAGS.config.eval_interval == 0: # not using the main process for evaluation because the main process use spawn method for dali, but hdf5 need fork method
            accelerator.wait_for_everyone()
            if FLAGS.config.eval_hdf5_path is not None:
                if accelerator.process_index==accelerator.num_processes-1:
                    discriminator.eval()
                    with torch.no_grad():
                        eval_info = evaluate_dataset(FLAGS.config.eval_hdf5_path, discriminator, FLAGS.config.eval_batch_size)
                        eval_info = dist.send_object_list([eval_info], 0)
                    discriminator.train()
                if accelerator.is_main_process:
                    eval_info = [None]
                    dist.recv_object_list(eval_info, accelerator.num_processes-1)
                    eval_info = eval_info[0]
                    wandb_log({"eval_"+ k: v for k, v in eval_info.items()}, i+1)
            accelerator.wait_for_everyone()
        
        
if __name__ == "__main__":
    app.run(main)