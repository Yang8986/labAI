import argparse
import importlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
sys.path.append(str(Path(__file__).parent.parent))
from labMLlib.train.predictor import Predictor
from labMLlib.Loss.lossDict import lossDict
from labMLlib.Optim.optimizerDict import optimDict
from labMLlib.utils.memOptim import free_memory
from labMLlib.utils.functions import predictDataTypeConvert

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cfg",
    type=str,
    help="config file for predictor (.json) path",
)
opt = parser.parse_args()
config_file_path = opt.cfg
config = json.load(open(config_file_path,"r",encoding="utf-8"))

device = (
    ("cuda" if torch.cuda.is_available() else "cpu")
    if config["device"] == "auto"
    else config["device"]
)
model = torch.load(
    config["predictStructurePath"],
    map_location=torch.device(device),
)
checkpoint = torch.load(
    config["predictWeightPath"],
    map_location=torch.device(device),
)
model.load_state_dict(checkpoint["model_state_dict"])
if device == "cuda":
    free_memory([checkpoint], debug=True)
datasetModule = importlib.import_module(
    f"labMLlib.datasets.{config['datasetType']}"
)
datasetCls = getattr(datasetModule, config["datasetType"])

predictor = Predictor(
    model=model,
    criterion=lossDict()[config["criterionType"]],
    dataloader=DataLoader(
        datasetCls(**config["datasetConfig"])
    ),
    mask=(
        None
        if not config["predictMaskPath"]
        else np.load(config["predictMaskPath"])
    ),
    device=config["device"],
    use_tqdm=False,
    save_dir=config["predictOutputDirPath"],
    vmin=config["vmin"],
    vmax=config["vmax"],
)
predictor.set_pbar()
predictor.loop()