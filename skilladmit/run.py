"""Arms, lifelong streams, Learn-then-Test selection, metrics and bootstrap CIs. Writes results/metrics.json.

python3 -m skilladmit.run [--seeds 0,1,2] [--n 1000]   (small: --n 60 --seeds 0)
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np
from scipy.stats import binom
from sklearn.manifold import TSNE
from skilladmit.data import episode, load
from skilladmit.dm import A, DM, FEATS, Env, Router, oracle_label
from skilladmit.ltt import floor, ltt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
ALPHAS, DELTA, HEAD = (0.02, 0.05, 0.1), 0.1, 0.05
LA = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98)
LR = (0.3, 0.5, 0.7, None)  # None = rewrite_desc disabled
TAU = tuple(round(t, 3) for t in np.arange(0.0, 0.951, 0.025))
DM_GRID = [("dm", a, r) for a in LA for r in LR]
SIM_GRID = [("sim", t) for t in TAU]
ARMS = [("always",), ("reject_all",), ("oracle",)] + SIM_GRID + DM_GRID
SPLITS = {"train": 0, "calib": 1, "test": 2, "test_iid": 3}
G = {}


def split_pool(pool, seed):
    idx = np.random.default_rng([seed, 99]).permutation(len(pool))
    k = len(pool) // 3
    return {"train": [pool[i] for i in idx[:k]], "calib": [pool[i] for i in idx[k:2 * k]],
            "test": [pool[i] for i in idx[2 * k:]], "test_iid": [pool[i] for i in idx[k:2 * k]]}


MIXES = {"mixed": dict(n_new=12, n_copy=6, n_blend=6), "real_only": dict(n_new=24, n_copy=0, n_blend=0)}


def stream(seed, split, i):
    mix = MIXES["mixed" if split == "train" else G.get("mix", "mixed")]
    return episode(G["pools"][split], np.random.default_rng([seed, SPLITS[split], i]), **mix)


def decide(env, c, arm, dm):
    """One admission decision. Returns (decision, skill to publish or None, info)."""
    if arm[0] == "always":
        return "admit", c, {}
    if arm[0] == "reject_all":
        return "reject", None, {}
    if arm[0] == "sim":
        s1 = float((env.L @ env.r.skill(c).T).max())
        return ("reject", None, {"s1": s1}) if s1 > arm[1] else ("admit", c, {"s1": s1})
    if arm[0] == "judge":
        sims = (env.L @ env.r.skill(c).T).toarray().ravel()
        pa = G["judge"].p_admit(c, [env.lib[j] for j in np.argsort(-sims)[:5]])
        return ("admit", c, {"p_admit": pa}) if pa >= arm[1] else ("reject", None, {"p_admit": pa})
    x, rw = env.features(c)
    if arm[0] == "oracle":
        d = oracle_label(env, c, rw)
        return d, {"admit": c, "rewrite_desc": rw, "reject": None}[d], {"s1": x[0]}
    p = dm.proba(x)
    info = {"s1": x[0], "p": [round(v, 4) for v in p]}
    if p[0] >= arm[1]:
        return "admit", c, info
    if rw is not None and arm[2] is not None and p[2] >= arm[2]:
        p2 = dm.proba(env.features(rw)[0])
        info["p_rw"] = [round(v, 4) for v in p2]
        if p2[0] >= arm[1]:
            return "rewrite_desc", rw, info
    return "reject", None, info


def simulate(base, cands, arm, dm, trace=False):
    env = Env(G["router"], base, cands)
    ev = ~env.probe
    steps, cnt = [], dict(admit=0, reject=0, rewrite_desc=0, synth_admitted=0, real_rejected=0)
    for c in cands:
        d, s, info = decide(env, c, arm, dm)
        cnt[d] += 1
        cnt["synth_admitted"] += bool(s is not None and c["synth"])
        cnt["real_rejected"] += bool(s is None and not c["synth"])
        if s is not None:
            env.add(s)
        ok = env.correct()
        po = (env.prev_ok & ev).sum()
        st = dict(risk=float((env.prev_ok & ev & ~ok).sum() / max(1, po)), acc=float((ok & ev).sum() / ev.sum()), lib=len(env.lib))
        if trace:
            st.update(name=c["name"], desc=c["desc"], kind=kind(c), decision=d, published=s and s["desc"], **info)
        steps.append(st)
    ok, po = env.correct(), env.prev_ok & ev
    out = dict(risk=float((po & ~ok).sum() / max(1, po.sum())), broken=int((po & ~ok).sum()), prev_ok=int(po.sum()),
               acc=float((ok & ev).sum() / ev.sum()), lib=len(env.lib), **cnt)
    if trace:
        out["steps"] = steps
        out["base"] = [dict(name=s["name"], desc=s["desc"], repo=s["repo"]) for s in base]
    return out, steps


def kind(c):
    return "real" if not c["synth"] else ("synth_blend" if "+" in c["id"] else "synth_copy")


def worker_init(seed, dm_state, mix="mixed"):
    pool = load()
    G.update(mix=mix, pools=split_pool(pool, seed), router=Router(pool), dm=dm_state and DM.from_state(dm_state))


def run_episode(args):
    seed, split, i, arms = args
    base, cands = stream(seed, split, i)
    res = {}
    for arm in arms:
        out, steps = simulate(base, cands, arm, G["dm"])
        res[arm] = (out, [(s["risk"], s["acc"], s["lib"]) for s in steps])
    return res


def train_rows(args):
    seed, i = args
    base, cands = stream(seed, "train", i)
    env, rows = Env(G["router"], base, cands), []
    for c in cands:
        x, rw = env.features(c)
        y = oracle_label(env, c, rw)
        rows.append((x.tolist(), y, kind(c)))
        s = {"admit": c, "rewrite_desc": rw, "reject": None}[y]
        if s is not None:
            env.add(s)
    return rows


def agg(results, arm):
    o = [r[arm][0] for r in results]
    f = {k: float(np.mean([x[k] for x in o])) for k in o[0]}
    f["pooled_risk"] = sum(x["broken"] for x in o) / max(1, sum(x["prev_ok"] for x in o))
    return f


def boot(a, b, n=2000, seed=0):
    d = np.asarray(a) - np.asarray(b)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def key(arm):
    return "/".join(str(v) for v in arm)


def run_seed(seed, n, n_train, procs, mix="mixed"):
    t0 = time.time()
    G.clear(); worker_init(seed, None)
    with Pool(procs, worker_init, (seed, None)) as p:
        rows = [r for rr in p.map(train_rows, [(seed, i) for i in range(n_train)]) for r in rr]
    X, y = np.array([r[0] for r in rows]), [r[1] for r in rows]
    dm = DM().fit(X, y)
    pred = [A[int(np.argmax(dm.proba(x)))] for x in X]
    train = dict(n_rows=len(rows), labels={a: y.count(a) for a in A}, train_acc=float(np.mean([a == b for a, b in zip(pred, y)])),
                 coef=dict(zip(A, [dict(zip(FEATS, map(float, w))) for w in dm.W])), sec=round(time.time() - t0, 1))
    out = dict(seed=seed, train=train, dm=dm.state(), pools={k: len(v) for k, v in G["pools"].items()})
    res = {}
    with Pool(procs, worker_init, (seed, dm.state(), mix)) as p:
        for split in ("calib", "test", "test_iid"):
            t = time.time()
            res[split] = p.map(run_episode, [(seed, split, i, ARMS) for i in range(n)], chunksize=4)
            out.setdefault("sec", {})[split] = round(time.time() - t, 1)
    out["grid"] = {split: {key(a): agg(res[split], a) for a in ARMS} for split in res}
    out["curves"] = {split: {key(a): np.mean([r[a][1] for r in res[split]], 0).round(5).tolist()
                             for a in [("always",), ("reject_all",), ("oracle",)] + SIM_GRID + DM_GRID} for split in ("test",)}
    out["per_alpha"] = {}
    for alpha in ALPHAS:
        sel = {}
        for name, grid in (("skilladmit", DM_GRID), ("sim_ltt", SIM_GRID)):
            cal = out["grid"]["calib"]
            valid = ltt(grid, lambda a: [(cal[key(a)]["risk"], n)], [alpha], DELTA)
            sel[name] = max((a for a, _ in valid), key=lambda a: cal[key(a)]["acc"]) if valid else ("reject_all",)
            sel[name + "_n_valid"] = len(valid)
        cal = out["grid"]["calib"]
        sel["sim_acc"] = max(SIM_GRID, key=lambda a: cal[key(a)]["acc"])
        sel["sim_size"] = min(SIM_GRID, key=lambda a: abs(cal[key(a)]["lib"] - cal[key(sel["skilladmit"])]["lib"]))
        sel["always"], sel["reject_all"], sel["oracle"] = ("always",), ("reject_all",), ("oracle",)
        row = {"alpha": alpha, "configs": {k: key(v) for k, v in sel.items() if not k.endswith("_n_valid")},
               "n_valid": {k[:-8]: v for k, v in sel.items() if k.endswith("_n_valid")}, "methods": {}}
        for m in ("skilladmit", "always", "sim_ltt", "sim_acc", "sim_size", "reject_all", "oracle"):
            a = sel[m]
            r = {s: out["grid"][s][key(a)] for s in ("calib", "test", "test_iid")}
            npo = sum(x[a][0]["prev_ok"] for x in res["test"])
            r["test_risk_upper95_at_alpha"] = float(binom.ppf(0.975, npo, alpha) / npo)
            r["inside_binomial_95"] = r["test"]["pooled_risk"] <= r["test_risk_upper95_at_alpha"]
            if m != "skilladmit":
                acc = lambda arm: [x[arm][0]["acc"] for x in res["test"]]
                r["acc_gap_skilladmit_minus"], r["acc_gap_ci95"] = boot(acc(sel["skilladmit"]), acc(a), seed=seed)
            row["methods"][m] = r
        out["per_alpha"][str(alpha)] = row
    out["ltt_floor_alpha"] = floor(n, len(DM_GRID), DELTA)
    out["sec"]["total"] = round(time.time() - t0, 1)
    return out, res


def traces(seed, dm_state, cfg, k=24):
    """Step-by-step test streams for the dashboard, with a 2-D t-SNE of every skill in the stream."""
    worker_init(seed, dm_state)
    eps = []
    for i in range(k):
        base, cands = stream(seed, "test", i)
        e = dict(id=i, arms={})
        for name, arm in cfg.items():
            e["arms"][name] = simulate(base, cands, arm, G["dm"], trace=True)[0]
        texts = [(s["name"], s["desc"]) for s in base] + [(c["name"], c["desc"]) for c in cands]
        V = G["router"].embed([f"{a}: {b}" for a, b in texts]).toarray()
        xy = TSNE(2, perplexity=12, init="pca", random_state=0).fit_transform(V)
        e["xy"] = (xy / np.abs(xy).max()).round(4).tolist()
        e["n_base"] = len(base)
        eps.append(e)
    return eps


def judge(seed, k):
    """Small-scale arm: the prompted judge at argmax (p_admit >= 0.5) on the first k test streams of one seed."""
    from skilladmit.dm import Judge
    M = json.load(open(os.path.join(RES, "metrics.json")))
    S = next(s for s in M["seeds"] if s["seed"] == seed)
    worker_init(seed, S["dm"])
    G["judge"], t0 = Judge(), time.time()
    cfg = S["per_alpha"][str(HEAD)]["configs"]
    arms = {"judge": ("judge", 0.5)} | {m: parse(cfg[m]) for m in ("skilladmit", "always", "sim_ltt", "oracle")}
    rows = []
    for i in range(k):
        base, cands = stream(seed, "test", i)
        rows.append({m: simulate(base, cands, a, G["dm"])[0] for m, a in arms.items()})
    out = dict(seed=seed, n_episodes=k, model=G["judge"].MODEL, revision=G["judge"].REVISION, threshold=0.5,
               n_prompts=len(G["judge"].cache), sec=round(time.time() - t0, 1), methods={})
    for m in arms:
        o = [r[m] for r in rows]
        out["methods"][m] = {q: float(np.mean([x[q] for x in o])) for q in o[0]}
        out["methods"][m]["pooled_risk"] = sum(x["broken"] for x in o) / max(1, sum(x["prev_ok"] for x in o))
        if m != "skilladmit":
            out["methods"][m]["acc_gap_skilladmit_minus"], out["methods"][m]["acc_gap_ci95"] = boot(
                [r["skilladmit"]["acc"] for r in rows], [r[m]["acc"] for r in rows])
    json.dump(out, open(os.path.join(RES, "judge.json"), "w"), indent=1)
    print(json.dumps(out["methods"]["judge"]), out["sec"], "s", file=sys.stderr)


def parse(k):
    return tuple(None if v == "None" else (float(v) if v[0].isdigit() else v) for v in k.split("/"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    ap.add_argument("--mix", default="mixed", choices=MIXES)
    ap.add_argument("--out", default=None)
    ap.add_argument("--judge", type=int, default=0, help="run only the prompted-judge arm on this many test streams")
    a = ap.parse_args()
    if a.judge:
        return judge(int(a.seeds.split(",")[0]), a.judge)
    seeds = [int(s) for s in a.seeds.split(",")]
    a.out = a.out or os.path.join(RES, "metrics.json" if a.mix == "mixed" else f"metrics_{a.mix}.json")
    M = dict(mix=a.mix, stream=MIXES[a.mix], n_episodes=a.n, n_train_episodes=a.n_train, delta=DELTA, alphas=ALPHAS, head_alpha=HEAD,
             grid=dict(lambda_admit=LA, lambda_rewrite=LR, tau=TAU), cpu=os.cpu_count(), seeds=[])
    for s in seeds:
        out, _ = run_seed(s, a.n, a.n_train, a.procs, a.mix)
        print(f"seed {s} done in {out['sec']['total']} s", file=sys.stderr, flush=True)
        M["seeds"].append(out)
        head = out["per_alpha"][str(HEAD)]
        for m, r in head["methods"].items():
            print(f"  {m:11s} {head['configs'][m]:16s} test risk {r['test']['risk']:.4f} acc {r['test']['acc']:.4f} "
                  f"lib {r['test']['lib']:.1f} gap {r.get('acc_gap_skilladmit_minus', 0):+.4f} {r.get('acc_gap_ci95', '')}",
                  file=sys.stderr, flush=True)
    h = M["seeds"][0]["per_alpha"][str(HEAD)]["configs"]
    cfg = {m: parse(h[m]) for m in ("skilladmit", "always", "sim_ltt", "oracle")}
    if a.mix == "mixed":
        M["traces"] = traces(seeds[0], M["seeds"][0]["dm"], cfg)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(M, open(a.out, "w"))
    print("wrote", a.out, file=sys.stderr)


if __name__ == "__main__":
    main()
