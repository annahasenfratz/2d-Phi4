#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, wasserstein_distance
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'perfect_blocking_upsampling/runs/lam1p0/paired_nativeL32_blockL16_reupscaleL32_N1000_20260721'
def obs(x):
 m=x.mean((1,2));p2=(x*x).mean((1,2));p4=(x**4).mean((1,2));nn=(x*np.roll(x,-1,1)+x*np.roll(x,-1,2)).mean((1,2));diag=(x*np.roll(np.roll(x,-1,1),-1,2)).mean((1,2));two=(x*np.roll(x,-2,1)+x*np.roll(x,-2,2)).mean((1,2));act=(1-x*x+.5*x**4-2*.340301*(x*np.roll(x,-1,1)+x*np.roll(x,-1,2))).mean((1,2));ft=np.fft.fft2(x,axes=(1,2));gp=.5*(np.abs(ft[:,1,0])**2+np.abs(ft[:,0,1])**2)/(x.shape[1]**2)
 return {'action':act,'phi2':p2,'phi4':p4,'kurtosis':p4/np.maximum(p2*p2,1e-12),'NN':nn,'diag':diag,'2nn':two,'m':m,'m2':m*m,'m4':m**4,'G_pmin_avg':gp}
def save(fig,name):fig.tight_layout();fig.savefig(RUN/'plots'/f'{name}.pdf');fig.savefig(RUN/'plots'/f'{name}.png',dpi=180);plt.close(fig)
def main():
 (RUN/'plots').mkdir(exist_ok=True);z=np.load(RUN/'paired_fields.npz');a,b=obs(z['native']),obs(z['upscaled']);keys=['action','phi2','phi4','kurtosis','NN','diag','2nn','m','m2','m4','G_pmin_avg']
 fig,axs=plt.subplots(3,4,figsize=(16,10));
 for ax,k in zip(axs.flat,keys):
  lo,hi=min(a[k].min(),b[k].min()),max(a[k].max(),b[k].max());bins=np.linspace(lo,hi,51);ax.hist(a[k],bins,density=True,histtype='step',lw=1.8,label='native');ax.hist(b[k],bins,density=True,histtype='step',lw=1.8,label='upscaled');ax.axvline(np.quantile(a[k],.05),c='k',ls=':',lw=.8);ax.axvline(np.quantile(a[k],.95),c='k',ls=':',lw=.8);ax.set_title(f'{k}: KS={ks_2samp(a[k],b[k]).statistic:.3f}');ax.legend(fontsize=7)
 for ax in axs.flat[len(keys):]:ax.axis('off')
 save(fig,'native_vs_upscaled_contact_sheet')
 keys2=['action','phi2','phi4','NN','m2','G_pmin_avg'];fig,axs=plt.subplots(2,3,figsize=(14,7))
 for ax,k in zip(axs.flat,keys2):ax.hist(b[k]-a[k],bins=50,density=True,color='#b34a3c');ax.axvline(0,c='k',lw=1);ax.set_title(f'{k} paired difference')
 save(fig,'paired_difference_histograms')
 fig,axs=plt.subplots(2,3,figsize=(14,8))
 for ax,k in zip(axs.flat,keys2):
  ax.scatter(a[k],b[k],s=5,alpha=.35);lo=min(a[k].min(),b[k].min());hi=max(a[k].max(),b[k].max());ax.plot([lo,hi],[lo,hi],'k--',lw=.8);ax.set_title(f'{k}, r={np.corrcoef(a[k],b[k])[0,1]:.3f}');ax.set_xlabel('native');ax.set_ylabel('upscaled')
 save(fig,'native_vs_upscaled_scatter')
if __name__=='__main__':main()
