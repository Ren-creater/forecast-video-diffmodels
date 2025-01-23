import os
import glob
import pickle
import pandas as pd
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

from utils import *

BASE_DATA = "/rds/general/ephemeral/user/zr523/ephemeral/satellite/"
BASE_DIR = f"{BASE_DATA}metadata"

def get_satmaps(region, name):
    satmaps = {
        "region": region, 
        "name": name
    }
    
    ERA5_BASE_DIR = f"{BASE_DATA}era5"
    SKIP_FRAMES = 1
    region = region_to_abbv[region]
    name = name.replace(' ', '').lower()
    
    if region == "nio":
        map_x0, map_y0 = 78.662109, 20.344627 ; hs_length = 20
        if name == "kyarr":
            map_x0, map_y0 = 65.662109, 20.344627 ; hs_length = 20
        if name == "gulab-shaheen":
            map_x0, map_y0 = 72.662109, 20.344627 ; hs_length = 23
    
    if region == "wpo":
        map_x0, map_y0 = 125.068359, 12.597455 ; hs_length = 25
    
    if region == "aus":
        map_x0, map_y0 = 131.681641, -20.244696 ; hs_length = 25
        if name == "niran":
            map_x0, map_y0 = 149.681641, -20.244696 ; hs_length = 25
    
    if region == "wio":
        map_x0, map_y0 = 46.8691, -18.7669 ; hs_length = 25

    if region == "use":
        map_x0, map_y0 = -80.1918, 25.7617 ; hs_length = 22
        if name == "bonnie":
            map_x0, map_y0 = -100.1918, 2.0617 ; hs_length = 32

    if region == "usw":        
        map_x0, map_y0 = -103.074219, 20.550509 ; hs_length = 10   
        if name == "genevieve":
            map_x0, map_y0 = -106.074219, 15.550509 ; hs_length = 16
        
    map_bounds = get_bbox_square(map_x0, map_y0, hs_length)
    era5_nc_files = sorted(glob.glob(f'{ERA5_BASE_DIR}/data/nc/{name}/*.nc'))
    mc_era5, era5_map_bounds =  get_era5_map(era5_nc_files, map_bounds)

    satmaps["map_bounds"] = [float(x) for x in era5_map_bounds]
    satmaps["era5_fns"] = era5_nc_files
    satmaps["satmaps"] = []
    
    if region == "nio":
        IR108_BASE_DIR = f"{BASE_DATA}mosdac"
        h5_files = sorted(glob.glob(f"{IR108_BASE_DIR}/data/h5/{name}/*/*.h5"))
        for idx in range(0, len(h5_files), SKIP_FRAMES):
            h5_file = h5_files[idx]
            date = " ".join(h5_file.split('/')[-1].split('_')[1:3])
            date = datetime.strptime(date, "%d%b%Y %H%M")
            date = round_to_closest_hour(date)
            satmaps["satmaps"].append({"date": date, "ir108_fn": h5_file}) 

    if region in ["aus", "wpo"]:
        IR108_BASE_DIR = f"{BASE_DATA}himawari"
        hr_dirs = sorted(glob.glob(f"{IR108_BASE_DIR}/data/bz2/{name}/*/*"))
        for idx in range(0, len(hr_dirs), SKIP_FRAMES):
            hr_dir = hr_dirs[idx]
            date = " ".join(hr_dir.split("/")[-2:])
            date = datetime.strptime(date, "%Y-%m-%d %H%M")
            if len(glob.glob(hr_dir+"/*.bz2")) > 0:
                satmaps["satmaps"].append({"date": date, "ir108_fn": hr_dir})  

    if region == "wio":
        IR108_BASE_DIR = f"{BASE_DATA}msg"
        nat_files = sorted(glob.glob(f"{IR108_BASE_DIR}/data/native/{name}/*.nat"))
        for idx in range(0, len(nat_files), SKIP_FRAMES):
            nat_file = nat_files[idx]
            date = nat_file.split('/')[-1].split('-')[-2].split('.')[0][:10]
            date = datetime.strptime(date, "%Y%m%d%H")
            satmaps["satmaps"].append({"date": date, "ir108_fn": nat_file})  

    if region == "use":
        IR108_BASE_DIR = f"{BASE_DATA}goes_east"
        nc_files = sorted(glob.glob(f"{IR108_BASE_DIR}/data/nc/{name}/*/*.nc"))
        for idx in range(0, len(nc_files), SKIP_FRAMES):
            nc_file = nc_files[idx]
            day = nc_file.split('/')[-2]
            hr = nc_file.split('_')[4][8:10]
            date = " ".join([day, hr])
            date = datetime.strptime(date, "%Y-%m-%d %H")
            satmaps["satmaps"].append({"date": date, "ir108_fn": nc_file})  

    if region == "usw":
        IR108_BASE_DIR = f"{BASE_DATA}goes_west"
        nc_files = sorted(glob.glob(f"{IR108_BASE_DIR}/data/nc/{name}/*/*.nc"))
        for idx in range(0, len(nc_files), SKIP_FRAMES):
            nc_file = nc_files[idx]
            day = nc_file.split('/')[-2]
            hr = nc_file.split('_')[4][8:10]
            date = " ".join([day, hr])
            date = datetime.strptime(date, "%Y-%m-%d %H")
            satmaps["satmaps"].append({"date": date, "ir108_fn": nc_file})   
    
    hrs = np.array([np64_to_datetime(x.values) for x in mc_era5["time"]])
    #print(era5_nc_files)
    #print(mc_era5["time"])
    #print(hrs)
    #print(satmaps["satmaps"])
    for idx in range(len(satmaps["satmaps"])):
        try:
            era5_idx = np.where(hrs == satmaps["satmaps"][idx]["date"])[0][0]
            satmaps["satmaps"][idx]["era5_idx"] = era5_idx
        except Exception as e:            
            print(f"[{name.upper()}]: Processing error at {satmaps['satmaps'][idx]['date']}")
            print(e)
            del satmaps["satmaps"][idx]

    satmaps["count"] = len(satmaps["satmaps"])
    satmaps["satmaps"] = sorted(satmaps["satmaps"], key=lambda k: k['era5_idx'])

    return satmaps, f"{region}_{name}"

cyclones_path = "./list_of_cyclones.xlsx"
df = pd.read_excel(cyclones_path)
df = df.drop('Unnamed: 8', axis=1)
df = df.dropna()

os.makedirs(BASE_DIR, exist_ok=True)

for idx in range(len(df)):
    row = df.iloc[idx]
    region = row["Region"]
    name = row["Name"]
    satmaps, fn = get_satmaps(region, name)
    with open(f'{BASE_DIR}/{fn}.metadata', 'wb') as metadata_file:
        pickle.dump(satmaps, metadata_file)
    print(f"[{name.upper()}]\tMetadata processing completed")
