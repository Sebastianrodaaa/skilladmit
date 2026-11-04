"""Checks on the real skill pool, the typed DM, LTT and the committed results. No torch. python3 test_skilladmit.py"""
import json, os
import numpy as np
from skilladmit import run
from skilladmit.data import episode, load
from skilladmit.dm import A, DM, Env, Router
from skilladmit.ltt import floor, hb_pvalue, ltt

ROOT = os.path.dirname(os.path.abspath(__file__))
pool = load()


def test_pool_is_real_and_attributed():
    assert len(pool) >= 300
    for s in pool:
        assert not s["synth"] and s["repo"] and len(s["commit"]) == 40 and s["license"] in ("MIT", "Apache-2.0")
        assert len(s["queries"]) >= 4 and len(s["desc"].split()) >= 5
    owners = {}
    for s in pool:
        for q in s["queries"]:
            owners.setdefault(q.lower(), set()).add(s["id"])
    assert all(len(v) == 1 for v in owners.values()), "every routing query has exactly one gold skill"


def test_synth_is_flagged_and_owns_no_tasks():
    base, cands = episode(pool, np.random.default_rng(0))
    syn = [c for c in cands if c["synth"]]
    assert syn and all(c["id"].startswith("SYNTH:") and c["queries"] == [] and c["dup_of"] for c in syn)
    env = Env(Router(pool), base, cands)
    assert all(env.sid(c) < 0 for c in syn)


def test_dm_simplex():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 15))
    y = [A[i] for i in rng.integers(0, 3, 300)]
    dm = DM().fit(X, y)
    dm2 = DM.from_state(json.loads(json.dumps(dm.state())))
    for x in X[:50]:
        p = dm2.proba(x)
        assert p.shape == (3,) and np.all(p >= 0) and abs(p.sum() - 1) < 1e-9
        assert np.allclose(p, dm.proba(x))


def test_ltt_refuses_below_floor():
    n, grid = 1000, list(range(40))
    f = floor(n, len(grid), 0.1)
    assert ltt(grid, lambda c: [(0.0, n)], [f * 0.9], 0.1) == []
    assert len(ltt(grid, lambda c: [(0.0, n)], [f * 1.1], 0.1)) == len(grid)
    assert hb_pvalue(0.05, n, 0.02) == 1.0


def test_gate_never_changes_the_router():
    run.worker_init(0, None)
    base, cands = run.stream(0, "test", 0)
    env = Env(run.G["router"], base, cands)
    before = (env.arg.copy(), env.best.copy(), len(env.lib))
    for c in cands:
        env.features(c)
        run.decide(env, c, ("sim", 0.3), None)
    assert np.array_equal(before[0], env.arg) and np.array_equal(before[1], env.best) and before[2] == len(env.lib)


def test_committed_results():
    M = json.load(open(os.path.join(ROOT, "results", "metrics.json")))
    assert len(M["seeds"]) == 3 and M["n_episodes"] >= 1000
    for s in M["seeds"]:
        cal = s["grid"]["calib"]
        for alpha in M["alphas"]:
            row = s["per_alpha"][str(alpha)]
            valid = ltt(run.DM_GRID, lambda c: [(cal[run.key(c)]["risk"], M["n_episodes"])], [alpha], M["delta"])
            assert row["n_valid"]["skilladmit"] == len(valid)
            if valid:  # the selection is the most accurate certified config, recomputed from the calibration grid
                assert cal[row["configs"]["skilladmit"]]["acc"] == max(cal[run.key(c)]["acc"] for c, _ in valid)
            else:
                assert row["configs"]["skilladmit"] == "reject_all"
        h = s["per_alpha"][str(M["head_alpha"])]["methods"]
        assert h["always"]["test"]["risk"] > M["head_alpha"], "always-admit must show a measurable routing drop"
    both = [st for e in M["traces"] for a, b in zip(e["arms"]["skilladmit"]["steps"], e["arms"]["sim_ltt"]["steps"])
            for st in [a] if a["kind"] != "real" and a["decision"] == "reject" and b["decision"] == "reject"]
    assert both, "skilladmit refuses at least one SYNTH near-duplicate that the similarity baseline also refuses"


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"):
            f()
            print("ok", k)
