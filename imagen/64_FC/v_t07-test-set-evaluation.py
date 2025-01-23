import glob
import copy
import argparse
from functools import partial, partialmethod

from einops import rearrange

import sys
sys.path.append("../")
sys.path.append("../imagen/")
sys.path.append("../../dataproc/")

from helpers import *
from imagen_pytorch import Unet3D, Imagen, ImagenTrainer, NullUnet
from send_emails import *

seed_value = 42
torch.manual_seed(seed_value)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)

parser = argparse.ArgumentParser()
parser.add_argument('-run_name', help='Specify the run name (for eg. 64_FC_3e-4)')
args = parser.parse_args()

sys.stdout = open(f'TEST_METRICS_LOG_{datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}.log','wt')
print = partial(print, flush=True)
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

RUN_NAME = args.run_name
BASE_DIR = f"{BASE_HOME}/models/{RUN_NAME}/models/64_FC/"

print(f"Run name: {RUN_NAME}")

best_epoch_dict = {
    "64_FC_rot904_sep_3e-4": 180,
    "64_FC_rot904_3e-4": 240,
    "64_FC_3e-4": 235
}

unet1 = Unet3D(
    dim = 32,
    cond_dim = 1024,
    dim_mults = (1, 2, 4, 8),
    num_resnet_blocks = 3,
    layer_attns = (False, True, True, True),
)  

unets = [unet1]

class DDPMArgs:
    def __init__(self):
        pass
    
args = DDPMArgs()
args.batch_size = 8
args.image_size = 64 ; args.o_size = 64 ; args.n_size = 128 ;
args.continuous_embed_dim = 64*64*3*args.batch_size
args.dataset_path = f"{BASE_DATA}/satellite/dataloader/{args.o_size}_FC"
args.datalimit = False
args.lr = 3e-4
args.mode = "fc"

train_dataloader, test_dataloader = get_satellite_data(args)
_ = len(train_dataloader) ; _ = len(test_dataloader)

print("Dataloaders loaded.")

if '1k' in RUN_NAME:
    timesteps = 1000
else:
    timesteps = 250

imagen = Imagen(
    unets = unets,
    image_sizes = (64),
    timesteps = 250,
    cond_drop_prob = 0.1,
    condition_on_continuous = True,
    continuous_embed_dim = args.continuous_embed_dim,
)

metric_dict = {
    "kl_div": [],
    "rmse": [],
    "mae":  [],
    "psnr": [],
    "ssim": [],
    "fid": []
}

test_metric_dict = copy.deepcopy(metric_dict)
best_epoch = best_epoch_dict[RUN_NAME]
ckpt_trainer_path = f"{BASE_DIR}/ckpt_trainer_1_{best_epoch:03}.pt"
trainer = ImagenTrainer(imagen, lr=args.lr, verbose=False).cuda()
trainer.load(ckpt_trainer_path) 

for idx in range(len(test_dataloader)):
    print(f"Evaluating batch idx {idx} ...")
    
    batch_idx = test_dataloader.random_idx[idx]
    img_64, _, era5 = test_dataloader.get_batch(batch_idx)
    cond_embeds = era5.reshape(1, -1).float().cuda()
    ema_sampled_vid = imagen.sample(
            batch_size = 1,#img_64.shape[0],          
            cond_scale = 3.,
            continuous_embeds = cond_embeds,
            use_tqdm = False,
            video_frames = 8
        )

    ema_sampled_vid = ema_sampled_vid.squeeze(0)
    ema_sampled_images = rearrange(ema_sampled_vid, 'c t h w -> t c h w')
    
    y_true = img_64.cpu()
    y_pred = ema_sampled_images.cpu()
    metric_dict = calculate_metrics(y_pred, y_true)
    for key in metric_dict.keys():
        test_metric_dict[key].append(metric_dict[key])

print("individual")
print(test_metric_dict)
# Initialize a dictionary to store the averages
average_metric_dict = {}

# Iterate over each key in the dictionary
for key, values in test_metric_dict.items():
    # Calculate the average for the current key
    average_metric_dict[key] = sum(values) / len(values)

# Now average_metric_dict will contain the average values for each key
print("average")
print(average_metric_dict)

with open(f"{BASE_HOME}/models/{RUN_NAME}/metrics_test.pkl", "wb") as file:
    pickle.dump(test_metric_dict, file)

print(f'Evaluation completed.')

subject = f"[COMPLETED] Test Evaluation Metrics"
message_txt = f"""Metrics Evaluation Completed for {RUN_NAME}"""
send_txt_email(message_txt, subject)