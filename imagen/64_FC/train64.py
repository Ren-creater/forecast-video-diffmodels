######################
# Code for Diffusion Models
# Ref: https://github.com/dome272/Diffusion-Models-pytorch
# Code for Implementation of Imagen, Google's Text-to-Image Neural Network, in Pytorch 
# Ref: https://github.com/lucidrains/imagen-pytorch
######################

import argparse
import os
import copy
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from torch import optim
import logging
from torch.utils.tensorboard import SummaryWriter

from einops import rearrange

import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.append("../")
sys.path.append("../imagen/")

from helpers import *
from imagen_pytorch import Unet3D, Imagen, ImagenTrainer

parser = argparse.ArgumentParser()
parser.add_argument('-run_name', help='Specify the run name (for eg. 64_FC_3e-4)')
parser.add_argument('-mode', help='Specify the mode [execute, experiment]')
parser.add_argument('-epochs', help='Specify the number of epochs')
parser.add_argument('--no_ema', action='store_false', dest='use_ema', help='disable ema for training')
parser.add_argument('--no_two_stage', action='store_false', dest='img', help='disable training with single frames')
cmd_args = parser.parse_args()

RUN_NAME = cmd_args.run_name
BASE_DIR = f"{BASE_HOME}/models/{RUN_NAME}"

os.makedirs(BASE_DIR, exist_ok=True)
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", 
                    level=logging.INFO, 
                    datefmt="%I:%M:%S",
                    filename=f"{BASE_DIR}/run.log")

seed_value = 42
torch.manual_seed(seed_value)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)

mode = cmd_args.mode

MODE = mode.upper()

from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

unets, O_SIZE = run_name_info(RUN_NAME)

def train(args):
    setup_logging(args.run_name, BASE_DIR)
    device = args.device
    train_dataloader, test_dataloader = args.dataloaders ; random_batch_idx = [5]
    logger = SummaryWriter(os.path.join(f"{BASE_DIR}/runs", args.run_name))
    epoch = 0

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.device("cuda"))
        _, max_memory = torch.cuda.mem_get_info()
        max_memory = max_memory/(1000**3)        
        logging.info(f"GPU Name: {gpu_name}")
        logging.info(f"Max GPU Memory: {max_memory} GiB")
    else:
        logging.info("No GPU available.")
    
    k = 1
    trainer = ImagenTrainer(imagen, use_ema = args.use_ema, lr=args.lr, verbose=False).cuda()
    try:
        ckpt_path = os.path.join(f"{BASE_DIR}/models", args.run_name, f"ckpt_{k}.pt")
        ckpt_trainer_path = os.path.join(f"{BASE_DIR}/models", args.run_name, f"ckpt_trainer_{k}.pt")
        checkpoint = torch.load(ckpt_path)
        trainer.load(ckpt_trainer_path)        
        start_epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        logging.info(f"Resuming training from epoch: {start_epoch+1} for unet_{k}")  
        if (start_epoch+1) >= args.epochs: 
            logging.info(f"No more epochs to train for unet_{k}")
        epoch = start_epoch+1
    except FileNotFoundError:
        start_epoch = -1
        loss = None
        logging.info(f"Starting training from scratch for unet_{k}")
        epoch = 0
    
    for epoch in range(start_epoch+1, args.epochs):
        logging.info(f"Starting epoch {epoch}:")
        if not cmd_args.img or epoch > args.epochs // 8:
            logging.info("Starting training on 10 frames videos")
            train_dataloader.switch_to_vid()
            test_dataloader.switch_to_vid()
            train_dataloader.create_batches(args.batch_size, False)
            test_dataloader.create_batches(args.batch_size, False)

        if args.shuffle_every_epoch:
            _ = len(train_dataloader) ; train_dataloader.create_batches(args.batch_size, False)
        print(len(train_dataloader))
        print(len(test_dataloader))
        pbar = tqdm(train_dataloader)

        for i, (vid_cond, vid_64, era5) in enumerate(pbar):            
            cond_embeds = era5.reshape(era5.shape[0], -1).float().cuda()                        
            loss = trainer(vid_64,
                           cond_video_frames=vid_cond,
                           continuous_embeds=cond_embeds,
                           unet_number=k,
                           ignore_time=False)
            trainer.update(unet_number=k)
    
            pbar.set_postfix({f"MSE_{k}":loss})
            logger.add_scalar(f"MSE_{k}",loss, global_step=epoch*len(train_dataloader)+i)

        checkpoint = {
            'epoch': epoch,
            'loss': loss
        }

        logging.info(f"Completed epoch {epoch}.")
        
        if (MODE == "EXPERIMENT") or \
           (MODE == "EXECUTE" and ((epoch % 5 == 0) or (epoch == (args.epochs-1)))):
            trainer.save(os.path.join(f"{BASE_DIR}/models", args.run_name, f"ckpt_trainer_{k}.pt"))
            torch.save(checkpoint, os.path.join(f"{BASE_DIR}/models", args.run_name, f"ckpt_{k}.pt"))
            trainer.save(os.path.join(f"{BASE_DIR}/models", args.run_name, f"ckpt_trainer_{k}_{epoch:03}.pt"))
            torch.save(checkpoint, os.path.join(f"{BASE_DIR}/models", args.run_name, f"ckpt_{k}_{epoch:03}.pt"))
               
        if args.sample:
            logging.info(f"Starting sampling for epoch {epoch}:") ; _ = len(test_dataloader)               
            random_batch = test_dataloader.random_idx[random_batch_idx][0]
            vid_cond, vid, era5 = test_dataloader.get_batch(random_batch)
            
            cond_embeds = era5.reshape(1, -1).float().cuda()
            ema_sampled_vid = imagen.sample(
                        batch_size = vid.shape[0],#img_64.shape[0],          
                        cond_scale = 3.,
                        continuous_embeds=cond_embeds,
                        use_tqdm = False,
                        video_frames = vid.shape[2],
                        cond_video_frames=vid_cond
                )
            #ema_sampled_vid = ema_sampled_vid.squeeze(0)
            ema_sampled_images = rearrange(ema_sampled_vid, 'b c t h w -> (b t) c h w')
            vid = rearrange(vid, 'b c t h w -> (b t) c h w')
            save_images_v2(test_dataloader, vid, ema_sampled_images, os.path.join(f"{BASE_DIR}/results", args.run_name, f"{epoch}_ema.jpg"))
            logging.info(f"Completed sampling for epoch {epoch}.")
                
import argparse

class DDPMArgs:
    def __init__(self):
        pass
    
args = DDPMArgs()
args.run_name = RUN_NAME
args.use_ema = cmd_args.use_ema
args.epochs = int(cmd_args.epochs)
args.batch_size = 1
args.image_size = O_SIZE ; args.o_size = O_SIZE ; args.n_size = 128 ;
#changed from 4 to 3 below, and * args.batch_size
args.continuous_embed_dim = args.o_size*args.o_size*3*10
args.dataset_path = f"{BASE_DATA}/satellite/dataloader/{args.o_size}_FC"
args.device = "cuda"
args.lr = 3e-4
args.sample = False#True
args.datalimit = False
args.augment = False#True
args.mode = "fc"
args.shuffle_every_epoch = False
#args.region = region_to_abbv["Australia"]#"North Indian Ocean"]

args.dataloaders = get_satellite_data(args, "vid")
logging.info(f"Dataset loaded")

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

print(f"number of parameters: {get_n_params(unets[0])}")
imagen = Imagen(
    unets = unets,
    image_sizes = (args.image_size),
    timesteps = 250,
    cond_drop_prob = 0.1,
    condition_on_continuous = True,
    continuous_embed_dim = args.continuous_embed_dim,
)

train(args)
