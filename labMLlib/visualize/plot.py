import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
plt.switch_backend('agg')

class ResearchPlot:
    def __init__(
        self,
        i,
        pred: torch.Tensor,
        answer_sample: torch.Tensor,
        label: str,
        mask: np.ndarray = None,
        save_dir: str = ".",
        style: str = "default",
    ) -> None:
        self.i = i
        self.pred = pred
        self.answer_sample = answer_sample
        self.label = label
        self.display = (
            ((pred - answer_sample) / answer_sample).detach().cpu().numpy().squeeze()
        )
        self.mask = np.ones(pred.shape) if not mask else mask.reshape(*pred.shape)
        self.display = self.display * np.where(self.mask == 0, np.nan, self.mask)
        self.save_dir = save_dir
        self.pred = self.pred.detach().cpu().numpy().squeeze() * np.where(
            self.mask == 0, np.nan, self.mask
        )
        self.answer_sample = (
            self.answer_sample.detach().cpu().numpy().squeeze()
            * np.where(self.mask == 0, np.nan, self.mask)
        )
        os.makedirs(os.path.join(self.save_dir, self.label), exist_ok=True)
        plt.style.use(style)

    def draw_figure(self):
        # plotting display and fig
        fig, ax = plt.subplots()
        display = plt.imshow(self.display, cmap="bwr")
        fig.colorbar(display, ax=ax)
        plt.clim(-0.1, 0.1)
        plt.rcParams.update({"font.size": 15})
        plt.savefig(os.path.join(self.save_dir, self.label, f"{self.i}.png"))
        plt.close()

    def draw_compare(self):
        # plt create colorbar images comparision
        fig, axes = plt.subplots(nrows=1, ncols=2)
        pred_im, answer_sample_im = axes[0].imshow(
            self.pred, cmap="bwr_r", vmin=0.5, vmax=1.5
        ), axes[1].imshow(self.answer_sample, cmap="bwr_r", vmin=0.5, vmax=1.5)
        fig.colorbar(answer_sample_im, ax=axes, pad=0.05, shrink=0.5)

        for ax in axes:
            ax.axis("off")
        plt.rcParams.update({"font.size": 20})
        plt.savefig(os.path.join(self.save_dir, self.label, f"{self.i}_compare.png"))
        plt.close()
    
    def draw_scatter(self):
        # plt scatter
        fig = plt.figure(figsize=(12, 10))
        display = plt.scatter(
            self.answer_sample.ravel(),
            self.pred.ravel(),
            s=1,
            c=self.answer_sample.ravel(),
            vmin=0,
            vmax=10,
        )
        fig.tight_layout()
        plt.savefig(os.path.join(self.save_dir, self.label, f"{self.i}_scatter.png"))
        plt.close()

def draw_scatter_alt(i,label,pred,answer_sample,save_dir,vmin=0,vmax=10,method="datashader",log=False): 
    # fig, (ax,colorbar_ax) = plt.subplots(nrows=1,ncols=2,width_ratios=[0.85,0.03],dpi=500)
    c_list = [
        (0, '#ffffff'),
        (1e-20, '#440053'),
        (0.2, '#404388'),
        (0.4, '#2a788e'),
        (0.6, '#21a784'),
        (0.8, '#78d151'),
        (1, '#fde624'),
    ]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('custom_viridis', c_list, N=256)
    if method == "scatter_density":
        try:
            import mpl_scatter_density
            fig = plt.figure(dpi=500)
            ax = fig.add_subplot(2, 1, 1, projection='scatter_density')
            ax_diff = fig.add_subplot(2, 1, 2, projection='scatter_density')
            density = ax.scatter_density(answer_sample, pred, cmap=cmap)
            density_diff = ax_diff.scatter_density(answer_sample, ((pred-answer_sample)/answer_sample)*100, cmap=cmap)
            # ax.set_xscale("log")
            # ax.set_yscale("log")
            ax.set_xlim(vmin,vmax)
            ax.set_ylim(vmin,vmax)
            # ax_diff.set_xscale("log")
            ax_diff.set_xlim(vmin,vmax)
            # ax_diff.set_ylim(-2,2)
            fig.colorbar(density, ax=ax, label='Number of points per pixel')
            fig.colorbar(density_diff, ax=ax_diff, label='Number of points per pixel')
            fig.savefig(f"{save_dir}/{label}/{i}_scatter_density.png")
            fig.savefig(os.path.join(save_dir, label, f"{i}_scatter_density.png"))
        except ImportError:
            print("mpl_scatter_density is not installed. Fallback to gaussian_kde")
            method = "gaussian_kde"
    if method == "datashader":
        try:
            import datashader as ds
            from datashader.mpl_ext import dsshow,EqHistNormalize
            import pandas as pd
            fig, [(ax,colorbar_ax),(ax_diff,colorbar_ax_diff)] = plt.subplots(nrows=2,ncols=2,width_ratios=[0.95,0.03],dpi=300,figsize=(6,12))
            formula_x = "answer_sample"
            formula_y = "pred"
            
            df = pd.DataFrame(dict(x=eval(formula_x), y=eval(formula_y)))
            dsartist = dsshow(
            df,
            ds.Point("x", "y"),
            ds.count(),
            norm="log",
            aspect="auto",
            ax=ax,
            cmap="jet",
            )
            del df

            # df = pd.DataFrame(dict(x=answer_sample, y=((pred-answer_sample)/answer_sample)))   # 沒有考慮到answer_sample=0的情況
            formula_x_diff = "answer_sample"
            formula_y_diff = "(pred-answer_sample)"
            df = pd.DataFrame(dict(x=eval(formula_x_diff), y=eval(formula_y_diff)))   
            
            dsartist_diff = dsshow(
            df,
            ds.Point("x", "y"),
            ds.count(),
            norm="log",
            aspect="auto",
            ax=ax_diff,
            cmap="jet",
            )
            
            plt.colorbar(dsartist,cax=colorbar_ax)
            plt.colorbar(dsartist_diff,cax=colorbar_ax_diff)

            # ax.set_xlabel(formula_x.replace("answer_sample","actual value"),labelpad=10)
            # ax.set_ylabel(formula_y.replace("pred","prediction"),labelpad=10)
            # ax_diff.set_xlabel(formula_x_diff.replace("answer_sample","actual value"),labelpad=10)
            # ax_diff.set_ylabel(formula_y_diff.replace("answer_sample","actual value").replace("pred","prediction"),labelpad=10)

            ax.plot(np.linspace(vmin,vmax),np.linspace(vmin,vmax),linestyle="--",alpha=0.5)
            ax.plot(np.linspace(vmin,vmax),np.linspace(vmin,vmax),linestyle="--",alpha=0.5)
            ax.set_xlim(vmin,vmax)
            ax.set_ylim(vmin,vmax)

            ax_diff.plot(np.linspace(vmin,vmax),np.zeros_like(np.linspace(vmin,vmax)),linestyle="--",alpha=0.5)
            ax_diff.plot(np.linspace(vmin,vmax),np.zeros_like(np.linspace(vmin,vmax)),linestyle="--",alpha=0.5)
            ax_diff.set_xlim(vmin,vmax)
            ax_diff.set_ylim(-1,1)
            
            fig.savefig(os.path.join(save_dir, label, f"{i}_datashader.png"))
            plt.close()
        except ImportError:
            import traceback
            traceback.print_exc()
            print("probably datashader is not installed. Fallback to gaussian_kde")
            method = "gaussian_kde"
    if method == "gaussian_kde":
        try:
            from scipy.stats import gaussian_kde
            fig, (ax,colorbar_ax) = plt.subplots(nrows=1,ncols=2,width_ratios=[0.85,0.03],dpi=500)
            xy = np.vstack([pred,answer_sample])
            z = gaussian_kde(xy)(xy)
            idx = z.argsort()
            x, y, z = pred[idx], answer_sample[idx], z[idx]*len(pred)
            ax.plot(np.linspace(np.min(x),np.max(x)),np.linspace(np.min(y),np.max(y)),linestyle="--",alpha=0.5)
            scatter = ax.scatter(x, y, c=z, s=3,cmap=cmap)
            colorbar = fig.colorbar(scatter, cax=colorbar_ax, orientation='vertical', label="testing")
            ax.set_xscale("log")
            ax.set_yscale("log")
            fig.savefig(f"{save_dir}/{label}/{i}_datashader.png",bbox_inches='tight')
            plt.close()
        except ImportError:
            import traceback
            traceback.print_exc()
            print("scipy is not installed. Please install scipy")
    plt.close('all')

    if log:
        has_log = False
        with open(f"{save_dir}/log.txt","r") as f:
            if f"{method}" in f.read():
                has_log = True
        if not has_log:
            with open(f"{save_dir}/log.txt","a") as f:
                f.write(f"{i} {label} {method}\n")
                f.write(f"formula_x: {formula_x}, formula_y: {formula_y}\n")
                f.write(f"formula_x_diff: {formula_x_diff}, formula_y_diff: {formula_y_diff}\n\n")