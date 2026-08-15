import csv
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'perfect_blocking_upsampling/runs/lam1p0/fresh_disjoint_calibrated_empirical_validation_20260721'
def read(p):return list(csv.DictReader(open(p)))
def write(p,r):
 with open(p,'w',newline='') as f:w=csv.DictWriter(f,r[0].keys());w.writeheader();w.writerows(r)
def main():
 out=[]
 for L,Lc in ((32,16),(64,32)):
  rows=read(RUN/f'patch_radial_latent_scan_L{L}.csv');sg=.32/Lc;ps=Lc
  r=next(x for x in rows if int(x['patch_size'])==ps and abs(float(x['sigma_gamma'])-sg)<1e-12)
  r.update({'Lf':L,'Lc':Lc,'sigma_gamma_law':sg,'law':'0.32/Lc','mode':'one configuration-global z'});out.append(r)
 write(RUN/'volume_scaled_global_latent_metrics.csv',out)
 write(RUN/'radial_variance_scaling.csv',[{'Lc':x['Lc'],'Vc':x['Lc']**2,'sigma_gamma':x['sigma_gamma_law'],'sigma_times_Lc':x['sigma_gamma_law']*x['Lc']} for x in out])
 (RUN/'volume_scaled_latent_specification.md').write_text('Frozen law: sigma_gamma=0.32/Lc=0.64/Lf. Apply D01,D10 -> 0.97 exp(sigma_gamma z) D01,D10; D11 unchanged; one z per configuration. The 1/Lc scaling is 1/sqrt(Vc) for a volume-averaged coarse radial mode. L8->L16 was not evaluated because no compatible frozen empirical L8->L16 proposal/raw paired sample is available.\n')
if __name__=='__main__':main()
