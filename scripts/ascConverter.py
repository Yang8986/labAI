import sys
import os
import time
import configparser
import argparse
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from labMLlib.utils.ascPreprocess import search_files_by_extension, asc2npz, asc2png,ascGenMask


parser = argparse.ArgumentParser()
parser.add_argument(
    "--cfg",
    type=str,
    default="./settings/ascConverter/ascConverter.json",
    help="config file (.json) path",
)
opt = parser.parse_args()
t = time.time()
config_file_path = opt.cfg
config = json.load(open(config_file_path,"r",encoding="utf-8"))
# config.read(open(config_file_path,"r",encoding="utf-8"))
convert2NPZ = config["settings"]["convert2npz"]
convert2PNG = config["settings"]["convert2png"]
try:genMask=config["settings"]["genMask"]
except KeyError: genMask=True
filetype = config["settings"]["filetype"]
dataroot = config["settings"]["dataroot"]
target_folder = config["settings"]["target_folder"]
try:remove_pattern = config["settings"]["remove_pattern"]
except KeyError: remove_pattern = "(?!.*)" # matches nothing so nothing will be removed
num_workers = int(config["settings"]["num_workers"])
folders = []
os.makedirs(target_folder, exist_ok=True)
for key, folder in config["folders"].items():
    asc_list = search_files_by_extension(".asc", chkpath=os.path.join(dataroot, folder))
    if convert2NPZ:
        asc2npz(
            asc_list=asc_list,
            target_folder=target_folder,
            fn=folder + ".npz",
            remove_pattern=remove_pattern,
            verbose=False,
        )
    if genMask:
        #add mask remove pattern
        # remove_asc_list = [i for i in asc_list if remove_pattern not in i]
        # print(f"remove asc list: {remove_asc_list}")
        # ascGenMask(remove_asc_list,target_folder)
        ascGenMask(asc_list,target_folder)
        import numpy as np
        from PIL import Image
        if os.path.exists(os.path.join(target_folder,"mask.npy")):
            mask = np.load(os.path.join(target_folder,"mask.npy")).astype(np.uint8)
            print(mask.shape)
            print(np.sum(mask,dtype=np.int16))
            mask = Image.fromarray(mask*255)
            mask.save(os.path.join(target_folder,"mask.png"))
        else:
            print("mask.npy not found, probably something went wrong during mask generation.")
    if convert2PNG:
        asc2png(asc_path=asc_list, target_folder=target_folder, verbose=False)
time_use = time.time()-t
print(f"time used: {time_use}s")  
with open (os.path.join(target_folder,"time_used.txt"),"w") as f:
    f.write(f"{time_use}s")