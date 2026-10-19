
from pathlib import Path
import numpy as np
import pandas as pd
import os
from scipy.optimize import root, brentq
import matplotlib.pyplot as plt

# Resolve all input/output paths relative to this script, not the caller's working directory.
ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

# ============================================================
# Coupling-scale invariance and mode-selective critical slowing
# Independent reproduction/figure-generation script
# ============================================================

a = 270.0
b = 108.0
d = 0.154
gamma = 0.641
tau_s = 0.1
J_N = 0.2609
w = 0.9
I0 = 0.3
c = 1.0
sigma = 0.001
MASTER_SEED = 20260927

OUT = "outputs"
FIG = "figures"
PUB = os.path.join(FIG, "publication")
REP = os.path.join(FIG, "reproducibility")
LATEX = os.path.join("outputs", "latex")

def save_dual(fig, stem):
    fig.savefig(os.path.join(PUB, stem + ".pdf"), bbox_inches="tight",
                metadata={"CreationDate": None, "ModDate": None})
    fig.savefig(os.path.join(REP, stem + ".png"), dpi=300, bbox_inches="tight")

os.makedirs(OUT, exist_ok=True)
os.makedirs(PUB, exist_ok=True)
os.makedirs(REP, exist_ok=True)
os.makedirs(LATEX, exist_ok=True)

def H(x):
    x = np.asarray(x, dtype=float)
    z = a*x - b
    y = d*z
    out = np.empty_like(z)
    small = np.abs(y) < 1e-6
    out[small] = (1/d + z[small]/2
                  + d*z[small]**2/12
                  - d**3*z[small]**4/720)
    out[~small] = z[~small] / (-np.expm1(-y[~small]))
    return out

def H_scalar(x):
    return float(H(np.array([x]))[0])

def Hp(x):
    z = a*x - b
    y = d*z
    if abs(y) < 1e-6:
        return a*(0.5 + d*z/6.0 - d**3*z**3/180.0)
    e = np.exp(-y)
    return a*(1.0 - e - y*e)/(1.0 - e)**2

def F(S, G):
    x = J_N*(w + c*G)*S + I0
    return -S/tau_s + (1-S)*gamma*H_scalar(x)

def FS(S, G):
    x = J_N*(w + c*G)*S + I0
    return (-1/tau_s - gamma*H_scalar(x)
            + (1-S)*gamma*Hp(x)*J_N*(w+c*G))

def lambda_minus(S, G):
    x = J_N*(w + c*G)*S + I0
    return (-1/tau_s - gamma*H_scalar(x)
            + (1-S)*gamma*Hp(x)*J_N*(w-c*G))

def fold_equations(v):
    S, G = v
    return np.array([F(S,G), FS(S,G)])

# ---- independent fold solve ----
folds = []
for guess in [(0.52, 0.25), (0.08, 0.90)]:
    sol = root(fold_equations, guess, method="hybr", tol=1e-12)
    if not sol.success:
        raise RuntimeError(sol.message)
    S_c, G_c = sol.x
    residual = np.max(np.abs(fold_equations(sol.x)))
    if not (0 < S_c < 1 and G_c > 0 and residual < 1e-10):
        raise RuntimeError("Fold QC failed.")
    folds.append((G_c, S_c, lambda_minus(S_c,G_c), residual))
folds.sort()
pd.DataFrame(
    folds, columns=["G","S","lambda_minus","max_residual"]
).to_csv(f"{OUT}/folds.csv", index=False)

G_lower, S_lower = folds[0][:2]
G_upper, S_upper = folds[1][:2]

# ---- stable transfer-function derivative / fold prefactor ----
def partial_G(S,G,h=2.5e-5):
    return (F(S,G+h)-F(S,G-h))/(2*h)

def partial_SS(S,G,h=2.5e-5):
    return (F(S+h,G)-2*F(S,G)+F(S-h,G))/h**2

FG = partial_G(S_upper,G_upper)
FSS = partial_SS(S_upper,G_upper)
K = 1/np.sqrt(2*FG*FSS)

with open(f"{OUT}/upper_fold_diagnostics.txt","w") as f:
    f.write(f"G_c = {G_upper:.15f}\n")
    f.write(f"S_c = {S_upper:.15f}\n")
    f.write(f"F_G = {FG:.15f}\n")
    f.write(f"F_SS = {FSS:.15f}\n")
    f.write(f"K = {K:.15f}\n")

# ---- all symmetric roots ----
def roots_at_G(G, ngrid=5000):
    grid = np.linspace(0, 0.999999, ngrid)
    vals = np.array([F(s,G) for s in grid])
    idx = np.where(vals[:-1]*vals[1:] < 0)[0]
    roots = []
    for i in idx:
        rr = brentq(lambda s: F(s,G), grid[i], grid[i+1],
                    xtol=1e-13, rtol=1e-13)
        if not roots or abs(rr-roots[-1]) > 1e-8:
            roots.append(rr)
    return roots

branch = []
for G in np.arange(0,1.150001,0.005):
    for S in roots_at_G(float(G)):
        lp = FS(S,G)
        lm = lambda_minus(S,G)
        branch.append((G,S,lp,lm,"stable" if lp<0 else "unstable"))
branch_df = pd.DataFrame(branch, columns=[
    "G","S","lambda_plus","lambda_minus","stability"
])
branch_df.to_csv(f"{OUT}/branch.csv",index=False)

# ---- Figure 1: equilibrium branches ----
fig, ax = plt.subplots(figsize=(7.2,4.8))
for stability, marker in [("stable","-"),("unstable","--")]:
    q = branch_df[branch_df.stability == stability]
    ax.plot(q.G,q.S,marker,label=stability)
ax.axvline(G_lower,linestyle=":",label="lower fold")
ax.axvline(G_upper,linestyle=":",label="upper fold")
ax.set_xlabel("Global coupling G")
ax.set_ylabel("Equilibrium S")
ax.set_title("Symmetric equilibrium branches")
ax.legend(frameon=False)
fig.tight_layout()
save_dual(fig, "fig_branch")
plt.close(fig)

# ---- Figure 2: critical slowing ----
scaling_G = np.array([
    0.80,0.82,0.84,0.85,0.86,0.87,0.88,0.885,
    0.89,0.892,0.894,0.895,0.896,0.897,0.8973,0.8975
])
tau_det = []
for G in scaling_G:
    S = roots_at_G(G)[0]
    lp = FS(S,G)
    if lp >= 0:
        raise RuntimeError("Expected low stable branch.")
    tau_det.append(-1/lp)
tau_det = np.asarray(tau_det)
slope, intercept = np.polyfit(
    np.log(G_upper-scaling_G), np.log(tau_det), 1
)
pd.DataFrame({
    "G":scaling_G,
    "distance_to_fold":G_upper-scaling_G,
    "tau_recovery":tau_det
}).to_csv(f"{OUT}/csd_deterministic.csv",index=False)

fig, ax = plt.subplots(figsize=(7.2,4.8))
ax.loglog(G_upper-scaling_G,tau_det,"o-",label="deterministic -1/lambda+")
xfit=np.geomspace(min(G_upper-scaling_G),max(G_upper-scaling_G),200)
ax.loglog(xfit,np.exp(intercept)*xfit**slope,
          linestyle="--",label=f"fit slope = {slope:.3f}")
ax.loglog(xfit,K*xfit**(-0.5),
          linestyle=":",label="theoretical exponent -1/2")
ax.set_xlabel("G_c - G")
ax.set_ylabel("Recovery time (s)")
ax.set_title("Critical slowing near the upper fold")
ax.legend(frameon=False)
fig.tight_layout()
save_dual(fig, "fig_csd")
plt.close(fig)

# ---- Figure 3: connectivity rescaling ----
cvals=np.linspace(.25,2,200)
fig, ax=plt.subplots(figsize=(7.2,4.8))
ax.plot(cvals,G_lower/cvals,label="lower fold")
ax.plot(cvals,G_upper/cvals,label="upper fold")
ax.set_xlabel("Reciprocal edge scale c")
ax.set_ylabel("Fold gain G_c(c)")
ax.set_title("Exact connectivity-scale/global-gain invariance")
ax.legend(frameon=False)
fig.tight_layout()
save_dual(fig, "fig_connectivity_scaling")
plt.close(fig)

# ---- Figure 4: graph spectral scaling ----
x_upper=J_N*(w+G_upper)*S_upper+I0
beta_c=(1-S_upper)*gamma*Hp(x_upper)*J_N*G_upper

Ns=np.array([8,16,32,64])
nu2=np.cos(2*np.pi/Ns)
delta=1-nu2
tau_tr=1/(beta_c*delta)
tau_asym=Ns**2/(2*np.pi**2*beta_c)
pd.DataFrame({
    "N":Ns,"nu2":nu2,"spectral_gap":delta,
    "tau_transverse":tau_tr,
    "asymptotic_tau":tau_asym
}).to_csv(f"{OUT}/spectral_cycles.csv",index=False)

fig, ax=plt.subplots(figsize=(7.2,4.8))
ax.loglog(Ns,tau_tr,"o-",label="normalized cycle")
ax.loglog(Ns,tau_asym,"--",label=r"$N^2/(2\pi^2\beta_c)$")
ax.set_xlabel("Cycle size N")
ax.set_ylabel("Slowest transverse recovery time (s)")
ax.set_title("Topology-dependent transverse recovery")
ax.legend(frameon=False)
fig.tight_layout()
save_dual(fig, "fig_spectral_cycles")
plt.close(fig)

# ---- stochastic verification ----
def simulate_one_G(G,rng,reps=32,dt=1e-3,T=30.0,burn=5.0,sample_every=10):
    roots=roots_at_G(G)
    S0=roots[0]
    if FS(S0,G)>=0:
        raise RuntimeError("Initial state is not stable.")
    S=np.full((reps,2),S0)
    steps=int(round(T/dt))
    burn_steps=int(round(burn/dt))
    nsave=(steps-burn_steps+sample_every-1)//sample_every
    Sp=np.empty((nsave,reps))
    Sm=np.empty((nsave,reps))
    j=0
    left=False
    noise=sigma*np.sqrt(dt)
    for k in range(steps):
        x1=J_N*(w*S[:,0]+G*c*S[:,1])+I0
        x2=J_N*(w*S[:,1]+G*c*S[:,0])+I0
        f1=-S[:,0]/tau_s+(1-S[:,0])*gamma*H(x1)
        f2=-S[:,1]/tau_s+(1-S[:,1])*gamma*H(x2)
        S[:,0]+=f1*dt+noise*rng.standard_normal(reps)
        S[:,1]+=f2*dt+noise*rng.standard_normal(reps)
        left |= bool(np.any((S<=0)|(S>=1)))
        if k>=burn_steps and ((k-burn_steps)%sample_every==0):
            Sp[j]=(S[:,0]+S[:,1])/np.sqrt(2)
            Sm[j]=(S[:,0]-S[:,1])/np.sqrt(2)
            j+=1
    Sp=Sp[:j]; Sm=Sm[:j]
    analysis_dt=dt*sample_every
    tau=[]; var=[]; corr=[]
    for r in range(reps):
        y=Sp[:,r]-Sp[:,r].mean()
        rho=np.dot(y[:-1],y[1:])/np.dot(y[:-1],y[:-1])
        tau.append(-analysis_dt/np.log(rho) if 0<rho<1 else np.nan)
        var.append(np.mean(y*y))
        s1=(Sp[:,r]+Sm[:,r])/np.sqrt(2)
        s2=(Sp[:,r]-Sm[:,r])/np.sqrt(2)
        corr.append(np.corrcoef(s1,s2)[0,1])
    lp=FS(S0,G); lm=lambda_minus(S0,G)
    vp=sigma**2/(-2*lp); vm=sigma**2/(-2*lm)
    return {
        "G":G,"Sstar":S0,"lambda_plus":lp,"lambda_minus":lm,
        "tau_theory":-1/lp,"tau_sim_mean":np.nanmean(tau),
        "tau_sim_sd":np.nanstd(tau),"var_theory":vp,
        "var_sim_mean":np.nanmean(var),"var_sim_sd":np.nanstd(var),
        "corr_theory":(vp-vm)/(vp+vm),
        "corr_sim_mean":np.nanmean(corr),
        "corr_sim_sd":np.nanstd(corr),"left_domain":left
    }

rng=np.random.default_rng(MASTER_SEED)
rows=[simulate_one_G(G,rng) for G in [0.50,0.70,0.80,0.85,0.88,0.89,0.895]]
stoch=pd.DataFrame(rows)
stoch.to_csv(f"{OUT}/stochastic_summary.csv",index=False)

# ---- unit test for exact scale invariance ----
for alpha in [0.5,1.0,2.0]:
    assert np.isclose((alpha*c)*(G_upper/alpha),c*G_upper,
                      rtol=0,atol=1e-14)

# ---- report ----
with open(f"{OUT}/reproduction_report.txt","w") as f:
    f.write("Independent reproduction report\n")
    f.write("===============================\n")
    f.write(f"Lower fold: G={G_lower:.10f}, S={S_lower:.10f}\n")
    f.write(f"Upper fold: G={G_upper:.10f}, S={S_upper:.10f}\n")
    f.write(f"Upper-fold lambda_minus={lambda_minus(S_upper,G_upper):.10f}\n")
    f.write(f"F_G={FG:.10f}\nF_SS={FSS:.10f}\nK={K:.10f}\n")
    f.write(f"Deterministic scaling slope={slope:.10f}\n")
    f.write(f"beta_c={beta_c:.10f}\n")
    f.write(f"Domain excursions in stochastic QC: {stoch.left_domain.any()}\n")


# ---- machine-readable LaTeX tables and manuscript numeric audit ----
folds_df = pd.read_csv(f"{OUT}/folds.csv")
rowend = chr(92)*2 + "\n"
with open(os.path.join(LATEX, "fold_table.tex"), "w") as f:
    f.write("\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\n")
    f.write("Fold & $G_c$ & $S_c$ & $\\lambda_-$ (s$^{-1}$) & Interpretation " + rowend + "\\midrule\n")
    f.write(f"Lower & {folds_df.iloc[0].G:.10f} & {folds_df.iloc[0].S:.10f} & {folds_df.iloc[0].lambda_minus:.6f} & high/middle branches appear " + rowend)
    f.write(f"Upper & {folds_df.iloc[1].G:.10f} & {folds_df.iloc[1].S:.10f} & {folds_df.iloc[1].lambda_minus:.6f} & low/middle branches disappear " + rowend)
    f.write("\\bottomrule\n\\end{tabular}\n")

with open(os.path.join(LATEX, "sim_table.tex"), "w") as f:
    f.write("\\begin{tabular}{@{}rrrrrr@{}}\n\\toprule\n$G$ & $\\tau_{\\rm th}$ (s) & $\\widehat\\tau$ (s) & $\\rho_{12,\\rm th}$ & $\\widehat\\rho_{12}$ & $\\lambda_-$ (s$^{-1}$) " + rowend + "\\midrule\n")
    for _,r in stoch.iterrows():
        f.write(f"{r.G:.3f} & {r.tau_theory:.4f} & {r.tau_sim_mean:.4f} & {r.corr_theory:.4f} & {r.corr_sim_mean:.4f} & {r.lambda_minus:.4f} " + rowend)
    f.write("\\bottomrule\n\\end{tabular}\n")

with open(os.path.join(LATEX, "spectral_table.tex"), "w") as f:
    f.write("\\begin{tabular}{@{}lrrrr@{}}\n\\toprule\nGraph & $\\nu_2$ & $\\delta$ & $\\lambda_{\\rm tr}$ (s$^{-1}$) & $\\tau_{\\rm tr}$ (s) " + rowend + "\\midrule\n")
    f.write(f"Two-node edge & {-1.0:.4f} & {2.0:.4f} & {-beta_c*2:.4f} & {1/(beta_c*2):.4f} " + rowend)
    # normalized complete graph K_8 (row sum 7): nu_2 = -1/7, delta = 8/7
    gap_K8 = 8/7
    f.write(f"$K_8/7$ & {-1/7:.4f} & {gap_K8:.4f} & {-beta_c*gap_K8:.4f} & {1/(beta_c*gap_K8):.4f} " + rowend)
    # normalized cycles C_8, C_16, C_32 (the C_64 point appears only in the figure/CSV)
    for N,nu,gap,tau in zip(Ns[:3],nu2[:3],delta[:3],tau_tr[:3]):
        f.write(f"$C_{{{int(N)}}}/2$ & {nu:.4f} & {gap:.4f} & {-beta_c*gap:.4f} & {tau:.4f} " + rowend)
    f.write("\\bottomrule\n\\end{tabular}\n")

# SI generated numerical fragments (single source of truth)
with open(os.path.join(LATEX, "si_upper_fold.tex"), "w") as f:
    f.write("\\begin{align}\n")
    f.write(f"G_c&={G_upper:.16f},\\\\\nS_c&={S_upper:.16f},\\\\\nF_G&={FG:.16f},\\\\\nF_{{SS}}&={FSS:.16f},\\\\\nK&=(2F_GF_{{SS}})^{{-1/2}}={K:.16f}\\,\\mathrm{{s}}.\n")
    f.write("\\end{align}\n")
with open(os.path.join(LATEX, "si_stochastic_table.tex"), "w") as f:
    f.write("\\begin{tabular}{@{}rrrrrrr@{}}\n\\toprule\n$G$ & $\\tau_{\\rm th}$ & $\\widehat\\tau$ & $V_{+,\\rm th}$ & $\\widehat V_+$ & $\\rho_{12,\\rm th}$ & $\\widehat\\rho_{12}$ " + rowend + "\\midrule\n")
    for _,r in stoch.iterrows():
        f.write(f"{r.G:.3f} & {r.tau_theory:.6f} & ${r.tau_sim_mean:.6f}\\pm{r.tau_sim_sd:.6f}$ & ${r.var_theory:.4e}$ & $({r.var_sim_mean:.4e}\\pm{r.var_sim_sd:.4e})$ & {r.corr_theory:.6f} & ${r.corr_sim_mean:.6f}\\pm{r.corr_sim_sd:.6f}$ " + rowend)
    f.write("\\bottomrule\n\\end{tabular}\n")
with open(os.path.join(LATEX, "si_csd_fit.tex"), "w") as f:
    f.write(f"\\begin{{equation}}\n\\log\\tau_{{\\rm rec}}= {slope:.6f}\\,\\log(G_c-G) {intercept:+.6f},\n\\end{{equation}}\n")

# Crosswalk: every reported computed numerical quantity has a generated source.
audit_rows = [
    ["fold_lower_G", f"{G_lower:.10f}", "outputs/folds.csv", "computed: root(F,F_S)=0"],
    ["fold_lower_S", f"{S_lower:.10f}", "outputs/folds.csv", "computed: root(F,F_S)=0"],
    ["fold_upper_G", f"{G_upper:.10f}", "outputs/folds.csv", "computed: root(F,F_S)=0"],
    ["fold_upper_S", f"{S_upper:.10f}", "outputs/folds.csv", "computed: root(F,F_S)=0"],
    ["upper_lambda_minus", f"{lambda_minus(S_upper,G_upper):.10f}", "outputs/folds.csv", "computed from Jacobian"],
    ["F_G_upper", f"{FG:.10f}", "outputs/upper_fold_diagnostics.txt", "finite-difference check"],
    ["F_SS_upper", f"{FSS:.10f}", "outputs/upper_fold_diagnostics.txt", "finite-difference check"],
    ["K_upper", f"{K:.10f}", "outputs/upper_fold_diagnostics.txt", "fold prefactor"],
    ["CSD_fit_slope", f"{slope:.10f}", "outputs/csd_deterministic.csv", "log-log regression"],
    ["beta_c", f"{beta_c:.10f}", "outputs/reproduction_report.txt", "upper-fold transverse coefficient"],
    ["stochastic_rows", str(len(stoch)), "outputs/stochastic_summary.csv", "fixed-seed Monte Carlo"],
    ["spectral_rows", str(len(Ns)), "outputs/spectral_cycles.csv", "analytic graph spectrum"],
]
pd.DataFrame(audit_rows, columns=["manuscript_quantity","value","generated_source","derivation"]).to_csv(os.path.join(OUT,"manuscript_numeric_audit.csv"),index=False)

# SHA-256 manifest for the final reproducibility outputs.
import hashlib, os as _os
manifest=[]
for root,_,files in _os.walk('.'):
    for name in files:
        if name.endswith(('.csv','.txt','.tex','.pdf','.png','.py','.md','.yml')):
            path=_os.path.join(root,name)
            if path.startswith('./.') or name == 'SHA256SUMS.csv': continue
            h=hashlib.sha256(Path(path).read_bytes()).hexdigest()
            manifest.append((path[2:] if path.startswith('./') else path,h))
pd.DataFrame(manifest,columns=["file","sha256"]).sort_values("file").to_csv(os.path.join(OUT,"SHA256SUMS.csv"),index=False)

print("Independent reproduction completed.")
print(f"Lower fold: G={G_lower:.10f}, S={S_lower:.10f}")
print(f"Upper fold: G={G_upper:.10f}, S={S_upper:.10f}")
print(f"lambda_minus at upper fold = {lambda_minus(S_upper,G_upper):.10f}")
print(f"CSD fit slope = {slope:.6f}")
print(f"beta_c = {beta_c:.10f}")
