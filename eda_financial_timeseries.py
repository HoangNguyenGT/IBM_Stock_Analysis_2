# eda_financial_timeseries.py
"""Exploratory data analysis and diagnostics for IBM, S&P 500, Gold, and the
7‑10 Year Treasury ETF (IEF).

Prerequisite
------------
`market_data_rebased.csv` must exist (created by *stock_fetch_rebase.py*).  It
contains daily adjusted prices for columns:
    IBM, SP500, GOLD, BOND
spanning 2002‑07‑26 onward.

Auto‑saved outputs
------------------
• ADF stationarity table (console)  
• `corr_heatmap.png`                – static correlation (returns)  
• `rolling_corr_21d.png`            – 21‑day rolling corr vs IBM  
• `ccf_<asset>_ibm.png`             – cross‑correlation stem plots  
• `granger_results.txt`             – p‑values (asset → IBM)  
• `var_ibm_coeff_t_heatmap.png`     – |t|-stats of VAR coefficients  
• `var_ibm_network.png`             – network graph (edge style reflects p‑value)  
• `var_ibm_network_edges.csv`       – edge list with exact p‑values  
• `johansen_results.txt`            – trace stats for cointegration
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller, ccf
from statsmodels.tsa.vector_ar.vecm import coint_johansen

warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------------------------------------------------------
# 1  Paths & parameters
# ------------------------------------------------------------------
CSV_PATH = Path("market_data_rebased.csv")
PLOT_DIR = Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

ROLL_WINDOW = 21     # trading days
MAX_CCF_LAG = 20     # ± lags for CCF plot
VAR_LAG_MAX = 15     # max order to search for VAR
SIG_T_CUT   = 2.0    # coefficient |t| significance
GRANGER_PCUT = 0.05  # p‑value cut‑off for solid edges

if not CSV_PATH.exists():
    raise FileNotFoundError("market_data_rebased.csv missing – run fetch script first.")

# ------------------------------------------------------------------
# 2  Load data & log‑returns
# ------------------------------------------------------------------
prices = pd.read_csv(CSV_PATH, parse_dates=["Date"], index_col="Date")
log_px = np.log(prices)
rets   = log_px.diff().dropna()

# ------------------------------------------------------------------
# 3  ADF + ACF/PACF COMPARISON GRIDS
# ------------------------------------------------------------------
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

def adf_acf_pacf_grid(df: pd.DataFrame, tag: str) -> None:
    """
    Make a 4×3 grid (line, ACF, PACF) for each column in *df*.
    Parameters
    ----------
    df  : DataFrame with Date index and one column per asset
    tag : "Levels" or "Returns" – used in y-axis label & file name
    """
    rows   = len(df.columns)
    y_text = {"Levels": "Log price",
              "Returns": "Log return"}[tag]

    fig, axes = plt.subplots(
        rows, 3,
        figsize=(11, 3.0 * rows),
        constrained_layout=True,
        gridspec_kw=dict(wspace=.25, hspace=.35)
    )
    if rows == 1:                       # ensure axes is 2-D
        axes = axes[None, :]

    for r, col in enumerate(df.columns):
        series = df[col].dropna()

        # ① Line plot -------------------------------------------------
        ax = axes[r, 0]
        ax.plot(series, lw=.8)
        ax.set_title(col, fontweight="bold", pad=6)
        ax.set_xlabel("Date" if r == rows-1 else "")
        ax.set_ylabel(y_text if r == 0 else "")

        # ② ACF -------------------------------------------------------
        ax = axes[r, 1]
        plot_acf(series, lags=40, ax=ax, alpha=.05, zero=False)
        ax.set_title("ACF (40 lags)", pad=6)
        ax.set_xlabel("Lag" if r == rows-1 else "")
        ax.set_ylabel("Corr" if r == 0 else "")
        ax.set_ylim(-1.1, 1.1)

        # ③ PACF + ADF verdict ---------------------------------------
        ax = axes[r, 2]
        plot_pacf(series, lags=40, ax=ax, alpha=.05, zero=False, method="ywm")
        p_adf   = adfuller(series)[1]
        verdict = "Stationary" if p_adf < .05 else "Non-stationary"
        ax.set_title(f"PACF (40)   ADF p = {p_adf:.3f}\n{verdict}",
                     fontsize=9, pad=6)
        ax.set_xlabel("Lag" if r == rows-1 else "")
        ax.set_ylabel("Part-Corr" if r == 0 else "")
        ax.set_ylim(-1.1, 1.1)

    fig.suptitle(f"{tag} — Series · ACF · PACF", fontsize=14, y=.995)

    fig.tight_layout()                # pack everything first
    fig.subplots_adjust(top=0.92)     # now carve out 8 % space on top
    fig.suptitle(f"{tag} — Series · ACF · PACF", fontsize=14)

    outfile = PLOT_DIR / f"acf_pacf_{tag.lower()}.png"
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f"Saved {outfile.name}")

# ---- build and save the two comparison figures -------------------
adf_acf_pacf_grid(log_px, "Levels")    # log-prices
adf_acf_pacf_grid(rets,   "Returns")   # daily log-returns

# ------------------------------------------------------------------
# 4  Correlation heat‑map & rolling correlations
# ------------------------------------------------------------------
# static correlation
heat = rets.corr()
heat.to_csv("corr_matrix.csv")
plt.figure(figsize=(6,5))
sns.heatmap(heat, annot=True, vmin=-1, vmax=1, cmap="coolwarm")
plt.title("Correlation – Daily log‑returns")
plt.tight_layout(); plt.savefig(PLOT_DIR/"corr_heatmap.png", dpi=300); plt.close()

# rolling correlations vs IBM
import matplotlib.dates as mdates

ROLL = ROLL_WINDOW                       # you can change this
assets = ["SP500", "GOLD", "BOND"]
roll = (
    rets.rolling(ROLL)
        .corr(rets["IBM"])
        .drop(columns=["IBM"])
        .pipe(lambda x: x[assets])     # force column order
        .dropna()
)

# smooth with a rolling median so spikes don't dominate
smooth = roll.rolling(ROLL, center=True, min_periods=1).median()

fig, axes = plt.subplots(len(assets), 1, figsize=(10, 2.7*len(assets)),
                         sharex=True, constrained_layout=True)

for i, asset in enumerate(assets):
    ax = axes[i]
    # shaded quantile band (5–95 %)
    q_low  = roll[asset].rolling(ROLL, center=True).quantile(.05)
    q_high = roll[asset].rolling(ROLL, center=True).quantile(.95)
    ax.fill_between(roll.index, q_low, q_high, color="#d9d9d9", alpha=.4,
                    label=f"{ROLL}-d 5-95 % range")

    # smoothed median line
    ax.plot(smooth.index, smooth[asset], lw=1.4, label=f"{asset}")

    # horizontal guides
    ax.axhline(0.5,  ls="--", lw=.7, color="grey"); ax.text(roll.index[3], .52, ".5")
    ax.axhline(0,    ls="--", lw=.7, color="grey"); ax.text(roll.index[3], .02, "0")
    ax.axhline(-.5,  ls="--", lw=.7, color="grey"); ax.text(roll.index[3], -.48, "-.5")

    # crisis bands
    ax.axvspan("2007-10-01", "2009-04-01", color="#ffcccc", alpha=.3)
    ax.axvspan("2020-02-15", "2020-06-15", color="#ffcccc", alpha=.3)

    ax.set_ylabel("Corr")
    ax.set_title(f"{asset} vs IBM")

# x-axis beautification
axes[-1].xaxis.set_major_locator(mdates.YearLocator(base=2))
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
axes[-1].set_xlabel("Date")

fig.suptitle(f"{ROLL}-day rolling correlation vs IBM\n"
             "(band = 5-95 % quantile, line = rolling median)",
             fontsize=14, y=.995)

fig.tight_layout(pad=1.2)          # 1️⃣ pack everything
fig.subplots_adjust(top=0.90)      # 2️⃣ leave 10 % head-room

out = PLOT_DIR / f"rolling_corr_{ROLL}d.png"
fig.savefig(out, dpi=300)
plt.close(fig)
print(f"Saved {out.name}")
# ------------------------------------------------------------------
# 5  Cross‑correlation stem plots (asset ↔ IBM)
# ------------------------------------------------------------------
assets = ["SP500", "GOLD", "BOND"]
lags   = np.arange(-MAX_CCF_LAG, MAX_CCF_LAG + 1)

fig, axs = plt.subplots(len(assets), 1,
                        figsize=(7, 3 * len(assets)),
                        sharex=True)

for i, asset in enumerate(assets):
    pos  = ccf(rets[asset], rets["IBM"], adjusted=False)[:MAX_CCF_LAG + 1]
    neg  = ccf(rets["IBM"], rets[asset], adjusted=False)[1:MAX_CCF_LAG + 1][::-1]
    axs[i].stem(lags, np.concatenate([neg, pos]),
                basefmt=" ")
    axs[i].axhline(0, color="red", lw=.7)
    axs[i].set_ylabel("CCF")
    axs[i].set_title(f"{asset}  ↔  IBM", pad=6, fontsize=11)

axs[-1].set_xlabel("Lag (days, +ve = asset leads)")

# tidy spacing: tight_layout first, then carve out space for the super-title
fig.tight_layout(pad=1.2)
fig.subplots_adjust(top=0.92)        # leave 8 % of the height free
fig.suptitle("Cross-correlation (CCF) — Assets vs IBM",
             fontsize=14, weight="bold")

out = PLOT_DIR / "ccf_assets_vs_ibm.png"
fig.savefig(out, dpi=300)
plt.close(fig)
print(f"Saved {out.name}")

# ------------------------------------------------------------------
# 6  Fit VAR and build coefficient heat‑map for IBM equation
# ------------------------------------------------------------------
var_mod   = VAR(rets)
lag_order = var_mod.select_order(VAR_LAG_MAX).aic
var_res   = var_mod.fit(lag_order)

ibm_row   = var_res.names.index("IBM")
ibm_tvals = var_res.tvalues.iloc[ibm_row]
if "const" in ibm_tvals.index:
    ibm_tvals = ibm_tvals.drop("const")

rows, lags, tvals = [], [], []
for lbl, t in ibm_tvals.items():
    if ".L" in lbl:
        var, lag = lbl.split(".L"); rows.append(var); lags.append(int(lag)); tvals.append(abs(t))
ht_df = pd.DataFrame({"var": rows, "lag": lags, "t": tvals})
if not ht_df.empty:
    piv = ht_df.pivot(index="var", columns="lag", values="t").sort_index()
    plt.figure(figsize=(1.1*piv.shape[1], 3))
    sns.heatmap(piv, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={"label": "|t|"})
    plt.title("|t|-stats of VAR coefficients predicting IBM")
    plt.ylabel("Predictor"); plt.xlabel("Lag (days)")
    plt.tight_layout(); plt.savefig(PLOT_DIR/"var_ibm_coeff_t_heatmap.png", dpi=300); plt.close()

# ------------------------------------------------------------------
# 7  Lag-aware Granger network graph (no excluded_terms needed)
# ------------------------------------------------------------------
import networkx as nx

lag_order      = var_res.k_ar
SIG_T_CUT      = 1.0       # |t| threshold for "individual lag matters"
GRANGER_PCUT   = 0.05
MIN_WIDTH      = 0.4

G          = nx.DiGraph()
edge_rows  = []
G.add_node("IBM")

for asset in ["SP500", "GOLD", "BOND"]:
    # joint p-value (all lags)
    p_joint = var_res.test_causality("IBM", [asset]).pvalue

    # find lags whose individual |t| > threshold
    sig_lags = []
    for j in range(1, lag_order + 1):
        coef_name = f"{asset}.L{j}"
        if coef_name in var_res.tvalues.columns:
            t_val = abs(var_res.tvalues.loc["IBM", coef_name])
            if t_val > SIG_T_CUT:
                sig_lags.append(str(j))

    # build graph edge
    G.add_node(asset)
    G.add_edge(asset, "IBM",
               width=max(MIN_WIDTH, -np.log10(p_joint)),
               style="solid" if p_joint < GRANGER_PCUT else "dashed",
               label=",".join(sig_lags) or "none",
               p_joint=p_joint)
    edge_rows.append((asset, "IBM", p_joint, ";".join(sig_lags)))

# draw graph
pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(6,4))
nx.draw_networkx_nodes(G, pos, node_color="#A7C7E7", node_size=1200)

edge_styles = [d["style"]  for _,_,d in G.edges(data=True)]
edge_widths = [d["width"]  for _,_,d in G.edges(data=True)]
nx.draw_networkx_edges(G, pos, width=edge_widths, style=edge_styles,
                       arrowstyle="->", arrowsize=15)

nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
edge_labels = {(u,v): d["label"] for u,v,d in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                             font_size=8, label_pos=0.5)

plt.title("Lag-specific Granger links to IBM\n"
          "(width ∝ −log10 joint-p, label = lags with |t|>1)")
plt.axis("off")
plt.tight_layout()
plt.savefig(PLOT_DIR / "var_ibm_network.png", dpi=300)
plt.close()

pd.DataFrame(edge_rows,
             columns=["source","target","joint_p","sig_lags"]
            ).to_csv("var_ibm_network_edges.csv", index=False)

print("Saved var_ibm_network.png and var_ibm_network_edges.csv")


# ------------------------------------------------------------------
# 8  Johansen cointegration
# ------------------------------------------------------------------
jo = coint_johansen(log_px,                 # <- log-price DataFrame
                    det_order=0,            # 0 → no deterministic trend
                    k_ar_diff=1)            # lag-1 differences

trace_stats = jo.lr1                        # array length = n_series
crit_vals   = jo.cvt[:, 1]                  # 95 % column

# --- tidy into a DataFrame for convenience ---------------------------------
ranks = np.arange(len(trace_stats))         # 0, 1, …, n-1
df_trace = pd.DataFrame({"r": ranks,
                         "trace": trace_stats,
                         "crit95": crit_vals})
df_trace["signif"] = df_trace["trace"] > df_trace["crit95"]

# --- plot ------------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.bar(df_trace["r"],
        df_trace["trace"],
        color=df_trace["signif"].map({True: "#1f77b4", False: "#bbbbbb"}),
        alpha=.9,
        label="Trace stat")
plt.plot(df_trace["r"],
         df_trace["crit95"],
         color="red", lw=1.5, label="95 % critical value")

plt.xticks(ranks, [f"r ≤ {r}" for r in ranks])
plt.ylabel("Statistic value")
plt.title("Johansen trace test")
plt.legend()
plt.tight_layout()
plt.savefig("plots/johansen_trace.png", dpi=300)


print("\nEDA complete – see ./plots/ for graphics and current directory for tables.")