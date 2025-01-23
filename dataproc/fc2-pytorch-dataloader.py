import glob
import torch
import numpy as np
import pickle

from tqdm import tqdm

from utils import *

BASE_DIR = "/rds/general/user/zr523/home/researchProject/satellite/dataloader/64_FC"
c_dataloader_fns = glob.glob(BASE_DIR + "/*.dat")

test_set = pickle.load(open("/rds/general/user/zr523/home/researchProject/forecast-diffmodels/dataproc/test_set.pkl", "rb"))
train_dataloader = ModelDataLoader(batch_size=4, mode="fc", augment=False#True
                                   )
test_dataloader  = ModelDataLoader(batch_size=4, mode="fc", test=True)

for fn in tqdm(c_dataloader_fns):
    with open(fn, "rb") as file: 
        region, name = fn.split('/')[-1][:-4].split('_')
        print(name)
        print(file)
        if name in test_set[region]:
            test_dataloader.add_dataloader(pickle.load(file))
        else:
            train_dataloader.add_dataloader(pickle.load(file))

_ = len(train_dataloader) ; _ = len(test_dataloader)