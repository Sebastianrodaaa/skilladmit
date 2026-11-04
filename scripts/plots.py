"""Black-and-white plots of the real run, from results/metrics*.json."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES, OUT = ROOT / "results", ROOT / "docs" / "screenshots"
plt.rcParams.update({"figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black",
                     "text.color": "#f2f2f2", "axes.labelcolor": "#a3a3a3", "axes.edgecolor": "#3a3a3a",
                     "xtick.color": "#a3a3a3", "ytick.color": "#a3a3a3", "font.family": "monospace", "font.size": 9,
                     "axes.grid": True, "grid.color": "#1c1c1c", "legend.frameon": False})
NAMES = {"skilladmit": "skilladmit", "always": "always-admit", "sim_ltt": "sim-reject (ltt τ)",
         "sim_acc": "sim-reject (best-acc τ)", "sim_size": "sim-reject (size-matched τ)", "reject_all": "reject-all",
         "oracle": "oracle"}
MARK = dict(zip(NAMES, "o s ^ v D x *".split()))


def load(name):
    p = RES / name
    return json.load(open(p)) if p.exists() else None


def risk_vs_alpha(Ms):
    fig, ax = plt.subplots(1, len(Ms), figsize=(5.2 * len(Ms), 3.6), squeeze=False)
    for j, (title, M) in enumerate(Ms):
        a = ax[0, j]
        for i, alpha in enumerate(M["alphas"]):
            for k, m in enumerate(NAMES):
                ys = [s["per_alpha"][str(alpha)]["methods"][m]["test"]["risk"] for s in M["seeds"]]
                c = "#f2f2f2" if m in ("skilladmit", "always") else "#7a7a7a"
                style = dict(color=c) if MARK[m] == "x" else dict(facecolor=c if m == "skilladmit" else "none", edgecolor=c)
                a.scatter([i + (k - 3) * 0.08] * len(ys), ys, marker=MARK[m], s=22, label=NAMES[m] if i == 0 else None, **style)
            a.hlines(alpha, i - 0.4, i + 0.4, color="#f2f2f2", lw=1.1, linestyles="--")
        a.set_xticks(range(len(M["alphas"])), [f"α={x}" for x in M["alphas"]])
        a.set_ylabel("test misroute risk"); a.set_title(f"{title}: test risk per seed vs α (dashed)", loc="left", fontsize=9)
    ax[0, -1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout(); fig.savefig(OUT / "risk.png", dpi=140); plt.close(fig)


def frontier(M):
    s = M["seeds"][0]
    g, sel = s["grid"]["test"], s["per_alpha"][str(M["head_alpha"])]["configs"]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for pre, mk, lab in (("sim/", "^", "similarity-reject τ grid"), ("dm/", "o", "skilladmit λ grid")):
        ks = [k for k in g if k.startswith(pre)]
        ax.scatter([g[k]["risk"] for k in ks], [g[k]["acc"] for k in ks], marker=mk, s=16,
                   facecolor="none", edgecolor="#f2f2f2" if pre == "dm/" else "#7a7a7a", label=lab)
    for m, k in sel.items():
        if m in ("skilladmit", "always", "sim_ltt", "oracle", "reject_all"):
            ax.scatter(g[k]["risk"], g[k]["acc"], marker=MARK[m], s=70, color="#f2f2f2", zorder=3)
            dy = {"skilladmit": 10, "sim_ltt": -14}.get(m, 0)
            ax.annotate(f" {NAMES[m]}", (g[k]["risk"], g[k]["acc"]), xytext=(28 if dy else 6, dy), textcoords="offset points",
                        fontsize=7, color="#f2f2f2", arrowprops=dict(arrowstyle="-", color="#6b6b6b", lw=0.6) if dy else None)
    ax.axvline(M["head_alpha"], color="#f2f2f2", ls="--", lw=1)
    ax.set_xlabel("test misroute risk on previously-correct tasks"); ax.set_ylabel("test routing accuracy")
    ax.set_title(f"seed {s['seed']}: every grid config on held-out test streams, α={M['head_alpha']} dashed", loc="left", fontsize=9)
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "frontier.png", dpi=140); plt.close(fig)


def growth(M):
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    for m, ls in (("skilladmit", "-"), ("always", "--"), ("sim_ltt", ":"), ("oracle", "-.")):
        Y = np.array([s["curves"]["test"][s["per_alpha"][str(M["head_alpha"])]["configs"][m]] for s in M["seeds"]])
        r = Y[:, :, 0].mean(0)
        ax.plot(np.arange(1, len(r) + 1), r, color="#f2f2f2" if m == "skilladmit" else "#a3a3a3", ls=ls, label=NAMES[m])
    ax.axhline(M["head_alpha"], color="#6b6b6b", ls=(0, (2, 4)), lw=1)
    ax.set_xlabel("candidates offered so far"); ax.set_ylabel("mean misroute risk")
    ax.set_title(f"misroute risk as the library grows, test, mean of {len(M['seeds'])} seeds × {M['n_episodes']} streams",
                 loc="left", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(OUT / "growth.png", dpi=140); plt.close(fig)


def admissions(M):
    ms = ("skilladmit", "always", "sim_ltt", "sim_acc", "oracle")
    h = str(M["head_alpha"])
    st = M["stream"]
    n_real, n_syn = st["n_new"], st["n_copy"] + st["n_blend"]
    real_in = [np.mean([n_real - s["per_alpha"][h]["methods"][m]["test"]["real_rejected"] for s in M["seeds"]]) / n_real for m in ms]
    syn_in = [np.mean([s["per_alpha"][h]["methods"][m]["test"]["synth_admitted"] for s in M["seeds"]]) / max(1, n_syn) for m in ms]
    fig, ax = plt.subplots(figsize=(7.4, 3.3))
    y = np.arange(len(ms))
    ax.barh(y - 0.18, real_in, 0.34, color="#f2f2f2", label=f"real candidates published (of {n_real})")
    ax.barh(y + 0.18, syn_in, 0.34, color="black", edgecolor="#a3a3a3", hatch="////", label=f"SYNTH near-dups published (of {n_syn})")
    ax.set_yticks(y, [NAMES[m] for m in ms]); ax.invert_yaxis(); ax.set_xlim(0, 1)
    ax.set_title(f"what each gate lets in, test, α={h}, mean of {len(M['seeds'])} seeds", loc="left", fontsize=9)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "admissions.png", dpi=140); plt.close(fig)


def table(M, title):
    """Markdown table, mean ± sd over seeds, every method at every alpha, held-out test streams."""
    out = [f"### {title}: {len(M['seeds'])} seeds × {M['n_episodes']} test streams, δ={M['delta']}\n",
           "| α | method | config (seed 0) | misroute risk | ≤ binom 95% | accuracy | Δacc skilladmit − method [95% CI, per seed] | library | SYNTH published | real staged |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    ms = lambda f: (np.mean(f), np.std(f))
    for alpha in M["alphas"]:
        for m in NAMES:
            R = [s["per_alpha"][str(alpha)]["methods"][m] for s in M["seeds"]]
            r, a = ms([x["test"]["risk"] for x in R]), ms([x["test"]["acc"] for x in R])
            gap = "" if m == "skilladmit" else "; ".join(f"{x['acc_gap_skilladmit_minus'] * 100:+.2f} [{x['acc_gap_ci95'][0] * 100:+.2f}, {x['acc_gap_ci95'][1] * 100:+.2f}]" for x in R)
            out.append(f"| {alpha} | {NAMES[m]} | `{M['seeds'][0]['per_alpha'][str(alpha)]['configs'][m]}` | {r[0]:.4f} ± {r[1]:.4f} | "
                       f"{sum(x['inside_binomial_95'] for x in R)}/{len(R)} | {a[0]:.4f} ± {a[1]:.4f} | {gap} | "
                       f"{np.mean([x['test']['lib'] for x in R]):.1f} | {np.mean([x['test']['synth_admitted'] for x in R]):.2f} | "
                       f"{np.mean([x['test']['real_rejected'] for x in R]):.2f} |")
    return "\n".join(out) + "\n"


def main():
    M, R = load("metrics.json"), load("metrics_real_only.json")
    risk_vs_alpha([("real + SYNTH", M)] + ([("real only", R)] if R else []))
    frontier(M); growth(M); admissions(M)
    J = load("judge.json")
    md = table(M, "real + SYNTH candidates") + ("\n" + table(R, "real candidates only") if R else "")
    if J:
        md += (f"\n### prompted judge (small-scale: seed {J['seed']}, {J['n_episodes']} test streams, uncalibrated p(admit) ≥ "
               f"{J['threshold']}, {J['model']})\n\n| method | misroute risk | accuracy | Δacc skilladmit − method [95% CI] | library | "
               "SYNTH published | real staged |\n|---|---|---|---|---|---|---|\n")
        for m, x in J["methods"].items():
            gap = "" if m == "skilladmit" else f"{x['acc_gap_skilladmit_minus'] * 100:+.2f} [{x['acc_gap_ci95'][0] * 100:+.2f}, {x['acc_gap_ci95'][1] * 100:+.2f}]"
            md += f"| {m} | {x['risk']:.4f} | {x['acc']:.4f} | {gap} | {x['lib']:.1f} | {x['synth_admitted']:.2f} | {x['real_rejected']:.2f} |\n"
    (RES / "summary.md").write_text(md)
    print("wrote", *sorted(p.name for p in OUT.glob("*.png")), file=sys.stderr)


if __name__ == "__main__":
    main()
