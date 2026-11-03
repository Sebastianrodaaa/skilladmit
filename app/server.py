"""Local dashboard: replays the committed real run (results/metrics.json) and scores a pasted candidate skill live
against a real base library with the trained DM and the frozen router.

Nothing is published anywhere: a decision only changes what the page shows.
Note: one process, one lock around the shared router; fine for one local user, not a shared deployment.
"""
import json, sys, threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from skilladmit import run  # noqa: E402
from skilladmit.dm import A, FEATS, Env  # noqa: E402
from skilladmit.ltt import ltt  # noqa: E402

RES = ROOT / "results"
MAX_PASTE = 8000
state, lock = {}, threading.Lock()


def boot():
    M = json.load(open(RES / "metrics.json"))
    seeds = [{k: s[k] for k in ("seed", "train", "pools", "sec", "per_alpha", "ltt_floor_alpha")} |
             {"calib_dm": {k: v for k, v in s["grid"]["calib"].items() if k.startswith("dm/")},
              "calib_sim": {k: v for k, v in s["grid"]["calib"].items() if k.startswith("sim/")},
              "curves": s["curves"]["test"],
              "valid": {str(a): [run.key(c) for c, _ in ltt(run.DM_GRID, lambda c: [(s["grid"]["calib"][run.key(c)]["risk"],
                                                                                     M["n_episodes"])], [a], M["delta"])]
                        for a in M["alphas"]}} for s in M["seeds"]]
    extra = {}
    for name in ("metrics_real_only.json", "judge.json"):
        if (RES / name).exists():
            X = json.load(open(RES / name))
            extra[name.split(".")[0]] = X if "methods" in X else {
                "mix": X["mix"], "seeds": [{k: s[k] for k in ("seed", "per_alpha")} for s in X["seeds"]]}
    state["summary"] = {k: v for k, v in M.items() if k not in ("seeds", "traces")} | {"seeds": seeds, "extra": extra}
    state["traces"] = M["traces"]
    s0 = M["seeds"][0]
    run.worker_init(s0["seed"], s0["dm"])
    state["seed"] = s0["seed"]


def live(body):
    """Score one pasted candidate against the base library of trace stream `id` (real skills, held-out test pool)."""
    c = dict(id="pasted", name=str(body.get("name", ""))[:120].strip(), desc=str(body.get("desc", ""))[:2000].strip(),
             queries=[], synth=True)
    if not c["name"] or len(c["desc"].split()) < 3:
        return {"error": "need a name and a description of 3+ words"}
    a = run.parse(str(body.get("cfg", "")))
    if a[0] != "dm":
        return {"error": "cfg must be a dm/λa/λr config"}
    with lock:
        base, _ = run.stream(state["seed"], "test", int(body.get("id", 0)) % len(state["traces"]))
        env = Env(run.G["router"], base, [])
        x, rw = env.features(c)
        d, pub, info = run.decide(env, c, a, run.G["dm"])
        sims = (env.L @ env.r.skill(c).T).toarray().ravel()
        near = [dict(name=base[j]["name"], sim=round(float(sims[j]), 3)) for j in sims.argsort()[::-1][:5]]
    return dict(decision=d, p=info.get("p"), p_rw=info.get("p_rw"), rewritten=rw and rw["desc"],
                published=pub and pub["desc"], feats=dict(zip(FEATS, map(float, x))), near=near, A=A)


class H(BaseHTTPRequestHandler):
    def send(self, body, ctype="application/json", code=200):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        u = urlparse(self.path).path
        if u == "/":
            return self.send((ROOT / "app" / "index.html").read_bytes(), "text/html; charset=utf-8")
        if u == "/api/summary":
            return self.send(state["summary"])
        if u == "/api/traces":
            return self.send(state["traces"])
        self.send({"error": "not found"}, code=404)

    def do_POST(self):
        if urlparse(self.path).path != "/api/live":
            return self.send({"error": "not found"}, code=404)
        n = int(self.headers.get("Content-Length", 0))
        if not 0 < n <= MAX_PASTE:
            return self.send({"error": f"body 1..{MAX_PASTE} bytes"}, code=400)
        try:
            body = json.loads(self.rfile.read(n))
        except ValueError:
            return self.send({"error": "body must be JSON"}, code=400)
        r = live(body if isinstance(body, dict) else {})
        self.send(r, code=400 if "error" in r else 200)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
    boot()
    print(f"skilladmit dashboard on http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
