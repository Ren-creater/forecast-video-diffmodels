import glob
import copy
import argparse
from functools import partial, partialmethod

from einops import rearrange

import sys
sys.path.append("../")
sys.path.append("../imagen/")
sys.path.append("../../dataproc/")

from utils import sample
from helpers import *
from imagen_pytorch import Unet3D, Imagen, ImagenTrainer, NullUnet
from send_emails import *

seed_value = 42
torch.manual_seed(seed_value)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed_value)

parser = argparse.ArgumentParser()
parser.add_argument('-run_name', help='Specify the run name (for eg. 64_FC_3e-4)')
args = parser.parse_args()

sys.stdout = open(f'METRICS_LOG_{datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")}.log','wt')
print = partial(print, flush=True)
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

RUN_NAME = args.run_name
BASE_DIR = f"{BASE_HOME}/models/{RUN_NAME}/models/{RUN_NAME}/"

print(f"Run name: {RUN_NAME}")

ckpt_files = sorted(glob.glob(BASE_DIR + "ckpt_1_*"))
ckpt_trainer_files = sorted(glob.glob(BASE_DIR + "ckpt_trainer_1_*"))

unets, O_SIZE = run_name_info(RUN_NAME)

class DDPMArgs:
    def __init__(self):
        pass
continuous_embed_dim = 10

args = DDPMArgs()
args.batch_size = 1
args.image_size = O_SIZE ; args.o_size = O_SIZE ; args.n_size = 128 ;
args.continuous_embed_dim = args.o_size*args.o_size*3*continuous_embed_dim
args.dataset_path = f"{BASE_DATA}/satellite/dataloader/{args.o_size}_FC"
args.datalimit = False
args.mode = "fc"
args.lr = 3e-4#float(RUN_NAME.split('_')[-1])

train_dataloader, test_dataloader = get_satellite_data(args, "vid")
train_dataloader.switch_to_vid()
test_dataloader.switch_to_vid()
_ = len(train_dataloader) ; _ = len(test_dataloader)

if '1k' in RUN_NAME:
    timesteps = 1000
else:
    timesteps = 250

random_idx = [5]

metric_dict = {
    "kl_div": [],
    "rmse": [],
    "mae":  [],
    "psnr": [],
    "ssim": [],
    "fid": [],
    "fvd": [],
    "ssim2": [],
    "psnr2": [],
    "lpips": []
}

train_test_metric_dict = {
    "train": copy.deepcopy(metric_dict), 
    "test": copy.deepcopy(metric_dict)
}

for idx in range(len(ckpt_trainer_files)):
    ckpt_trainer_path = ckpt_trainer_files[idx]
    print(f'Evaluating {ckpt_trainer_path.split("/")[-1]} ...')

    for mode in ["train", "test"]:
        if mode == "train" : dataloader = train_dataloader
        elif mode == "test": dataloader = test_dataloader
    
        y_true, y_pred = sample(dataloader, RUN_NAME, random_idx[0], args, ckpt_trainer_path)  
        
        metric_dict = calculate_metrics(y_pred, y_true)
        for key in metric_dict.keys():
            train_test_metric_dict[mode][key].append(metric_dict[key])

with open(f"{BASE_HOME}/models/{RUN_NAME}/metrics.pkl", "wb") as file:
    pickle.dump(train_test_metric_dict, file)

print(f'Evaluation completed.')

subject = f"[COMPLETED] Evaluation Metrics"
message_txt = f"""Metrics Evaluation Completed for {RUN_NAME}"""
send_txt_email(message_txt, subject)