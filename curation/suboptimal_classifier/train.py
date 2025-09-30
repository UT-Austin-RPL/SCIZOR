from curation.suboptimal_classifier.dataset.dataset import AugActionDataset, ExternIterator 
from curation.suboptimal_classifier.discriminator.discriminator import Discriminator
from octo.octo.data.oxe import make_oxe_dataset_kwargs_and_weights
from octo.octo.utils.train_utils import Timer

from absl import flags, app
from ml_collections import config_flags
import os
import torch
from tqdm import tqdm
import wandb
from utils import wandb_log, wandb_init, Torch_Preallocater
from accelerate import Accelerator

FLAGS = flags.FLAGS

flags.DEFINE_string("name", "experiment", "Experiment name.")
flags.DEFINE_bool("debug", False, "Debug config (no wandb logging)")

config_dir = os.path.join(os.path.dirname(__file__), "config")
config_flags.DEFINE_config_file(
    "config",
    os.path.join(config_dir, "config.py"),
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

timer = Timer()

def create_dataset(config):
    extern_source = ExternIterator(FLAGS.config)
    aug_action_dataset = AugActionDataset(FLAGS.config, extern_source, **FLAGS.config.dali_kwargs)
    next(aug_action_dataset)
    return aug_action_dataset

def main(_):
    torch.manual_seed(FLAGS.config.seed)
    wandb_id = wandb_init(FLAGS.config, FLAGS.name, FLAGS.debug)
        
    discriminator = Discriminator(**FLAGS.config.discriminator)
    if "oxe_kwargs" in FLAGS.config.dataset_kwargs:
        # create dataset_kwargs_list from oxe_kwargs
        (
            FLAGS.config.dataset_kwargs["dataset_kwargs_list"],
            FLAGS.config.dataset_kwargs["sample_weights"],
        ) = make_oxe_dataset_kwargs_and_weights(
            **FLAGS.config.dataset_kwargs["oxe_kwargs"]
        )
        del FLAGS.config.dataset_kwargs["oxe_kwargs"]
        
    save_dir = FLAGS.config.save_folder
    if save_dir is not None:
        exp_dir = os.path.join(save_dir, wandb_id)
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)
    else:
        exp_dir = None
        
    optimizer = torch.optim.AdamW(discriminator.parameters(), **FLAGS.config.optimizer)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=FLAGS.config.scheduler.gamma)
    dataset = create_dataset(FLAGS.config)

    num_devices = torch.cuda.device_count()   
    discriminator = discriminator.to(device)
    discriminator.train()
    
    
    def train_step(discriminator:Discriminator, batch, optimizer, scheduler):
        with timer("datset"):
            images, actions, action_scores, task_desc = batch
            # For now, only take the first timestep of image, action, action_score
            images, actions, action_scores = images[:, 0], actions[:, 0], action_scores[:, 0]
            images, actions, action_scores = images.to(device), actions.to(device), action_scores.to(device)
            
        with timer("train"):
            result = discriminator.forward(images, task_desc, actions, action_scores)
            
            optimizer.zero_grad()
            result["loss"].backward()
            optimizer.step()
            
            scheduler.step()
            
        return result["loss"].item().detach().cpu().numpy()
    
    tqdm_bar = tqdm(range(FLAGS.config.num_steps))
    for i in tqdm_bar:
        batch = next(dataset)
        loss = train_step(discriminator, batch, optimizer, scheduler)
        dataset.skip(num_devices)
                
        tqdm_bar.set_postfix({f"loss_{FLAGS.config.discriminator.loss_fn_type}": loss})
        if (i+1) % FLAGS.config.log_interval == 0 and not FLAGS.debug:
            wandb_log({"loss": loss, "lr": optimizer.param_groups[0]["lr"], 'time': timer.get_average_times()}, i)
        
        if (i+1) % FLAGS.config.save_interval == 0 and exp_dir is not None:
            torch.save(discriminator.state_dict(), os.path.join(exp_dir, f"model_{i}.pth"))
        
        
if __name__ == "__main__":
    app.run(main)