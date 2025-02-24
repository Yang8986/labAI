from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import LogFormatter
import os, csv
from datetime import datetime
import numpy as np
import sys
from pathlib import Path
from labMLlib.visualize.plot import draw_scatter_alt


class Predictor:
    def __init__(
        self,
        model,
        criterion,
        dataloader: DataLoader,
        mask: None | np.ndarray = None,
        device="auto",
        # tqdm params
        use_tqdm=True,
        description_format="prediction loss:%.6f",
        position=0,
        # plot
        is_draw_figures=True,
        is_output_loss=True,
        save_dir="output",
        vmin=None,
        vmax=None,
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.dataloader = dataloader
        self.mask = mask
        # print("mask:",self.mask)
        # print("mask average:",np.mean(self.mask))
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
            if device == "auto"
            else device
        )
        self.use_tqdm = use_tqdm
        self.description_format = description_format
        self.position = position
        self.is_draw_figures = is_draw_figures
        self.is_output_loss = is_output_loss
        self.save_dir = save_dir
        self.vmin = vmin
        self.vmax = vmax
        self.time_used = 0
        self.model.to(self.device)
        self.model.eval()
        self.criterion.to(self.device)
        self.dataloader_iter = iter(dataloader)
        self.label_root = self.dataloader_iter._next_data()[0] if len(self.dataloader_iter._next_data()) == 3 else None

        self.current_label = None

        if isinstance(self.label_root, tuple):
            self.label_root = self.label_root[0]
        print(self.label_root)
        self.label_root = set(self.label_root.split(os.sep)) if self.label_root else ""
        for data in dataloader:
            if len(data) == 3:
                self.data_length = 3
                label = data[0][0]
                # print("label in dataloader: ",label)
                if isinstance(label, tuple):
                    label = label[
                        0
                    ]  # extract label from tuple because dataloader will change datatype of label from str to tuple.
                self.label_root &= set(label.split(os.sep))
            elif len(data) == 2:
                self.data_length = 2
                self.label_root = ""
                break
            else:
                raise NotImplementedError(
                    f"data length mismatch! Should be 2 or 3 but got {len(data)} instead!"
                )

    def __call__(
        self,
        i,
        data,
    ):
        if self.data_length == 3:
            label, data_sample, answer_sample = data
            if isinstance(label, tuple):
                label = label[
                    0
                ]  # extract label from tuple because dataloader will change datatype of label from str to tuple.
            for p in self.label_root:
                label = label.replace(p, "")
            label = str(Path(label))
            if label.startswith(os.sep):
                label = Path(label[1:])
            else:
                label = Path(label)
        elif self.data_length == 2:
            data_sample, answer_sample = data
            label = ""
        data_sample, answer_sample = data_sample.to(self.device), answer_sample.to(
            self.device
        )
        s = datetime.now()
        preds = self.model(data_sample)
        time_used = (datetime.now() - s).total_seconds()
        preds = preds.view(*answer_sample.shape)
        self.mask = (
            self.mask if isinstance(self.mask, np.ndarray) else np.ones(preds.shape)
        )
        preds *= torch.from_numpy(self.mask).to(self.device)
        answer_sample *= torch.from_numpy(self.mask).to(self.device)
        
        loss = self.criterion(preds, answer_sample)
        
        self.preds = preds.detach().cpu().numpy()

        if self.use_tqdm:
            self.pbar.set_description(self.description_format % loss)
        if self.is_draw_figures:
            self.draw_figures(i, preds, answer_sample, label)
        if self.is_output_loss:
            self.output_loss(i, preds, answer_sample, label, loss.detach().cpu().item(),time_used=time_used)
        return loss.detach().cpu().item(),time_used

    def draw_figures(
        self,
        i,
        pred,
        answer_sample,
        label,
    ):
        display = (
            ((pred - answer_sample) / answer_sample).detach().cpu().numpy().squeeze()
        )
        self.mask = self.mask.reshape(*display.shape)
        display *= np.where(self.mask == 0, np.nan, self.mask)

        # plotting display and fig
        fig, ax = plt.subplots()
        display = plt.imshow(display, cmap="bwr")
        fig.colorbar(display, ax=ax)
        plt.rcParams.update({"font.size": 15})
        plt.axis("off")
        os.makedirs(os.path.join(self.save_dir,label), exist_ok=True)
        plt.savefig(os.path.join(self.save_dir,label,f"{i}.png"))
        plt.close()

        # plt create colorbar images comparision
        fig, axes = plt.subplots(nrows=1, ncols=2)
        pred, answer_sample = pred.detach().cpu().numpy().squeeze() * np.where(
            self.mask == 0, np.nan, self.mask
        ), answer_sample.detach().cpu().numpy().squeeze() * np.where(
            self.mask == 0, np.nan, self.mask
        )
        
        if self.vmin == self.vmax:
            fig.set_size_inches(14, 5)
            # fig_min = min(np.min(pred),np.min(answer_sample))
            # fig_max = max(np.max(pred),np.max(answer_sample))
            fig_min = 0
            fig_max = 1.5
        else:
            fig_min = self.vmin
            fig_max = self.vmax

        pred_im, answer_sample_im = axes[0].imshow(
            pred, cmap="jet", vmin=fig_min, vmax=fig_max
        ), axes[1].imshow(answer_sample, cmap="jet", vmin=fig_min, vmax=fig_max)
        fig.colorbar(answer_sample_im, ax=axes, pad=0.05, shrink=0.5)
        # fig.colorbar(answer_sample_im, ax=axes, pad=0.05)

        for ax in axes:
            ax.axis("off")
        plt.rcParams.update({"font.size": 20})
        plt.savefig(f"{self.save_dir}/{label}/{i}_compare.png")
        plt.close()

        # plt diff
        fig, ax = plt.subplots()
        diff = pred - answer_sample
        y_len = len(diff)
        fig_min_diff = abs(np.ceil(np.min(diff)*100)/100)
        fig_max_diff = abs(np.ceil(np.max(diff)*100)/100)
        log_max_diff = np.max(diff)
        log_min_diff = np.min(diff)
        log_mean_diff = np.mean(diff)
        v = max(fig_min_diff, fig_max_diff)
        diff = plt.imshow(diff, cmap="bwr", vmin=-1, vmax=1)
        fig.colorbar(diff, ax=ax)
        plt.rcParams.update({"font.size": 15})
        os.makedirs(os.path.join(self.save_dir,label), exist_ok=True)
        plt.savefig(os.path.join(self.save_dir,label,f"{i}_diff.png"))
        plt.close()

        # with open(f"{self.save_dir}/{label}/{i}_diff.txt","w") as f:
        #     f.write(f"min diff: {log_min_diff}\n")
        #     f.write(f"max diff: {log_max_diff}\n")
        #     f.write(f"mean diff: {log_mean_diff}\n")

        # plt scatter
        fig = plt.figure(figsize=(12, 10))
        display = plt.scatter(
            answer_sample.ravel(),
            pred.ravel(),
            s=1,
            c=answer_sample.ravel(),
            vmin=fig_min,
            vmax=fig_max,
        )
        fig.tight_layout()
        os.makedirs(os.path.join(self.save_dir,label), exist_ok=True)
        plt.savefig(os.path.join(self.save_dir,label,f"{i}_scatter.png"))
        plt.close()

    def output_loss(
        self,
        i,
        pred,
        answer_sample,
        label,
        loss,
        time_used
    ):
        if not os.path.exists(os.path.join(self.save_dir,"loss.csv")):
            with open(os.path.join(self.save_dir,"loss.csv"),"w") as f:
                w = csv.writer(f, lineterminator="\n") 
                w.writerow(["event","number","loss","time_used"])
        with open(os.path.join(self.save_dir,"loss.csv"),"a") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow([label,i,loss,time_used])
        os.makedirs(os.path.join(self.save_dir,label), exist_ok=True)
        with open(f"{self.save_dir}/{label}/{i}_loss.txt","w") as f:
            f.write(f"loss: {loss}\n")

    # def loop(self):
    #     total_loss = 0
    #     avg_time_used = 0
    #     min_time_used = 1e6
    #     max_time_used = 0
    #     self.set_pbar()
    #     with open(f"{self.save_dir}/time_used.txt","w") as f:
    #         s = datetime.now()
    #         for i, data in self.pbar:
    #             s_single = datetime.now()
    #             loss = self.__call__(i, data)
    #             total_loss += loss
    #             time_used = (datetime.now() - s_single).total_seconds()
    #             min_time_used = min(min_time_used, time_used)
    #             max_time_used = max(max_time_used, time_used)
    #             avg_time_used += time_used
    #         avg_time_used /= len(self.dataloader)
    #         f.write(f"total time used: {datetime.now()-s}\n")
    #         f.write(f"average time used per prediction: {avg_time_used} seconds\n")
    #         f.write(f"min time used for single perdiction: {min_time_used} seconds\n")
    #         f.write(f"max time used for single perdiction: {max_time_used} seconds\n")
    #     return float(total_loss / len(self.dataloader))

    def loop(self):
        total_loss = 0
        self.set_pbar()
        x,y = np.array([]),np.array([])

        #save time
        with open(os.path.join(self.save_dir,"time.csv"),"w") as f:  
            w = csv.writer(f, lineterminator="\n") 
            w.writerow(["event","number","time"]) 

        # save log
        with open(os.path.join(self.save_dir,"log.txt"),"a") as f:
            f.write(f"start time: {datetime.now()}\n")
            f.write(f"device: {self.device}\n")
            f.write(f"model: {self.model}\n")
            f.write(f"shape of data: {self.dataloader.dataset[0][1].shape}\n")
            f.write(f"criterion: {self.criterion}\n")
            f.write(f"mask: {self.mask}\n")
            f.write(f"save_dir: {self.save_dir}\n")
            f.write(f"vmin: {self.vmin}\n")
            f.write(f"vmax: {self.vmax}\n")
            f.write(f"is_draw_figures: {self.is_draw_figures}\n")
            f.write(f"is_output_loss: {self.is_output_loss}\n")
            f.write(f"use_tqdm: {self.use_tqdm}\n")
            f.write(f"description_format: {self.description_format}\n")
            f.write(f"position: {self.position}\n")
            f.write(f"current_label: {self.current_label}\n")
            f.write(f"label_root: {self.label_root}\n")
            f.write(f"data_length: {self.data_length}\n\n")

        count = 0
        for i, data in self.pbar:
            loss, spend_time = self.__call__(i, data)
            if len(data) == 3:
                label, _, answer_sample = data
                if isinstance(label, tuple):
                    label = label[
                        0
                    ]  # extract label from tuple because dataloader will change datatype of label from str to tuple.
                    for p in self.label_root:
                        label = label.replace(p, "")
                    label = str(Path(label))
                    if label.startswith(os.sep):
                        label = Path(label[1:])
                    else:
                        label = Path(label)
            elif len(data) == 2:
                _, answer_sample = data
                label = ""
            else:
                raise NotImplementedError(
                    f"data length mismatch! Should be 2 or 3 but got {len(data)} instead!"
                )
            self.preds *= np.where(self.mask == 0, np.nan, self.mask) # 把mask裡面的0換成nan
            self.preds = self.preds.flatten()[~np.isnan(self.preds.flatten())] # 把nan去掉
            self.preds = np.where(self.preds<0,0,self.preds)
            answer_sample = answer_sample.detach().cpu().numpy()
            answer_sample *= np.where(self.mask == 0, np.nan, self.mask)
            answer_sample = answer_sample.flatten()
            answer_sample = answer_sample[~np.isnan(answer_sample)]
            answer_sample = np.where(answer_sample<0,0,answer_sample) # 把小於0的值換成0
            
            # when the event is diffrernt, reset the x and y to avoid the scatter plot is too large
            if label != self.current_label:
                self.current_label = label
                count = 0
                x = np.array([])
                y = np.array([])
            x = np.concatenate([x,self.preds.flatten()])
            y = np.concatenate([y,answer_sample])

            if self.use_tqdm:
                self.pbar.set_description("drawing figures...")

            # tmp_change
            if (self.mask!=np.ones(self.mask.shape)).any() or self.vmin==self.vmax:
                draw_scatter_alt(i, label, x, y, self.save_dir, method="datashader",vmax=1,log=True)
            else:
                print("mask is not set, scatter plot will not be drawn.")
            total_loss += loss

            with open(os.path.join(self.save_dir,"time.csv"),"a") as f:
                count+=1
                w = csv.writer(f, lineterminator="\n")
                w.writerow([label,count,spend_time])

        with open(os.path.join(self.save_dir,"loss.txt"),"a") as f:  
            f.write(f"total loss: {total_loss / len(self.dataloader)}\n")

        return float(total_loss / len(self.dataloader))

    def set_pbar(self):
        self.pbar = (
            tqdm(
                enumerate(self.dataloader),
                total=len(self.dataloader),
                position=self.position,
                leave=False,
            )
            if self.use_tqdm
            else enumerate(self.dataloader)
        )
        if self.use_tqdm:
            self.pbar.set_description(self.__class__.__name__)
