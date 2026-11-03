"""The live router (fixed name+description TF-IDF top-1), the library state, and the typed admission DM.

The DM is a System One classifier over the fixed option set A = (admit, reject, rewrite_desc). It reads the candidate's
name+description, the current library's metadata and the library's *probe* routing tasks (half of each admitted
skill's queries); it never sees eval tasks or the candidate's own queries.
"""
import re
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

A = ("admit", "reject", "rewrite_desc")
FEATS = ("s1", "s2", "gap12", "top3", "name_jac", "n_stolen", "frac_stolen", "frac_stolen_nn", "frac_attract",
         "max_gain", "has_rw", "rw_s1", "rw_n_stolen", "rw_frac_stolen", "log_words")
TOK = re.compile(r"[a-z0-9]+")


def text(s):
    return f"{s['name'].replace('-', ' ').replace('_', ' ')}: {s['desc']}"


class Router:
    """Progressive-disclosure router: cosine of the task to each skill's name+description, top-1. Fit once, frozen."""

    def __init__(self, pool):
        self.vec = TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), stop_words="english").fit(map(text, pool))
        self.cache = {}

    def embed(self, texts):
        return self.vec.transform(texts).tocsr()

    def skill(self, s):
        k = (s["name"], s["desc"])
        if k not in self.cache:
            self.cache[k] = self.embed([text(s)])
        return self.cache[k]


def content(t):
    return {w for w in TOK.findall(t.lower()) if w not in ENGLISH_STOP_WORDS}


def rewrite(c, nn):
    """rewrite_desc operator (deterministic stand-in for an author edit): drop description words shared with the
    nearest library skill's name+description. None if fewer than 4 content words would remain."""
    shared = content(text(nn))
    keep = [w for w in c["desc"].split() if not (content(w) and content(w) <= shared)]
    if len(content(" ".join(keep))) < 4 or len(keep) == len(c["desc"].split()):
        return None
    return dict(c, desc=" ".join(keep), rewritten=True)


class Env:
    """One episode: tasks of every real skill in the stream, the library, and current top-1 routing of every task."""

    def __init__(self, router, base, cands):
        self.r = router
        real = base + [c for c in cands if not c["synth"]]
        self.gid = {s["id"]: i for i, s in enumerate(real)}
        qs = [(q, self.gid[s["id"]], i % 2 == 0) for s in real for i, q in enumerate(s["queries"])]
        self.Q = router.embed([q for q, _, _ in qs])
        self.gold = np.array([g for _, g, _ in qs])
        self.probe = np.array([p for _, _, p in qs])
        self.best = np.full(len(qs), -1.0)
        self.arg = np.full(len(qs), -1)
        self.prev_ok = np.zeros(len(qs), bool)
        self.inlib = np.zeros(len(qs), bool)
        self.lib, self.L = [], None
        for s in base:
            self.add(s, mark=False)
        self.mark(self.inlib)

    def sid(self, s):
        return self.gid.get(s["id"], -2 - len(self.lib))  # SYNTH skills own no tasks

    def col(self, s):
        return (self.Q @ self.r.skill(s).T).toarray().ravel()

    def mark(self, m):
        self.prev_ok[m] = (self.arg == self.gold)[m]

    def add(self, s, mark=True):
        col, g = self.col(s), self.sid(s)
        up = col > self.best
        self.best[up], self.arg[up] = col[up], g
        v = self.r.skill(s)
        self.L = v if self.L is None else sparse.vstack([self.L, v]).tocsr()
        self.lib.append(s)
        own = self.gold == g
        self.inlib |= own
        if mark:
            self.mark(own)

    def correct(self):
        return self.arg == self.gold

    def _probe_stats(self, s):
        col = self.col(s)
        ok = self.probe & self.correct()
        win = col > self.best
        stolen = ok & win
        sims = (self.L @ self.r.skill(s).T).toarray().ravel()
        nn = int(sims.argmax())
        nn_ok = self.probe & (self.gold == self.sid(self.lib[nn])) & self.inlib
        pl = self.probe & self.inlib
        return sims, nn, dict(n_stolen=stolen.sum(), frac_stolen=stolen.sum() / max(1, ok.sum()),
                              frac_stolen_nn=(stolen & nn_ok).sum() / max(1, nn_ok.sum()),
                              frac_attract=(win & pl).sum() / max(1, pl.sum()),
                              max_gain=float((col - self.best)[pl].max()) if pl.any() else 0.0)

    def features(self, c):
        """Typed DM input for candidate c against the current library. Returns (x, rewritten candidate or None)."""
        sims, nn, st = self._probe_stats(c)
        top = np.sort(sims)[::-1]
        s2 = top[1] if len(top) > 1 else 0.0
        a, b = set(TOK.findall(c["name"].lower())), set(TOK.findall(self.lib[nn]["name"].lower()))
        rw = rewrite(c, self.lib[nn])
        if rw is not None:
            rs, _, rst = self._probe_stats(rw)
            r = (1.0, rs.max(), rst["n_stolen"], rst["frac_stolen"])
        else:
            r = (0.0, top[0], st["n_stolen"], st["frac_stolen"])
        x = [top[0], s2, top[0] - s2, top[:3].mean(), len(a & b) / max(1, len(a | b)), st["n_stolen"],
             st["frac_stolen"], st["frac_stolen_nn"], st["frac_attract"], st["max_gain"], *r,
             np.log1p(len(c["desc"].split()))]
        return np.array(x, float), rw

    def effect(self, c):
        """Oracle effect with gold labels (training labels and the oracle reference arm only): tasks of the library
        that c would misroute (harm, probe+eval) and c's own tasks it would win (value)."""
        col, own = self.col(c), self.gold == self.sid(c)
        win = col > self.best
        return int((self.correct() & win).sum()), int((own & win).sum())


def oracle_label(env, c, rw):
    """Net-benefit label: the option with the most (own tasks won - library tasks misrouted); reject (0) wins ties."""
    net = {"reject": 0, "admit": (lambda h, v: v - h)(*env.effect(c))}
    if rw is not None:
        net["rewrite_desc"] = (lambda h, v: v - h)(*env.effect(rw))
    return max(net, key=lambda k: (net[k], -("reject", "admit", "rewrite_desc").index(k)))


class DM:
    """Multinomial logistic regression over the typed features; proba() returns a simplex over A."""

    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        self.lr = LogisticRegression(max_iter=5000, C=1.0).fit(self.sc.transform(X), [A.index(v) for v in y])
        W = np.zeros((3, X.shape[1])); b = np.full(3, -1e9)
        W[self.lr.classes_], b[self.lr.classes_] = self.lr.coef_, self.lr.intercept_  # absent class -> p = 0
        self.W, self.b, self.mu, self.sd = W, b, self.sc.mean_, self.sc.scale_
        return self

    def proba(self, x):
        z = self.W @ ((x - self.mu) / self.sd) + self.b
        e = np.exp(z - z.max())
        return e / e.sum()

    def state(self):
        return dict(W=self.W.tolist(), b=self.b.tolist(), mu=self.mu.tolist(), sd=self.sd.tolist(), feats=FEATS, A=A)

    @classmethod
    def from_state(cls, d):
        m = cls()
        m.W, m.b, m.mu, m.sd = (np.array(d[k]) for k in ("W", "b", "mu", "sd"))
        return m


class Judge:
    """Prompted LLM judge baseline (Qwen2.5-0.5B-Instruct on CPU): p(admit) from the next-token logits of
    'admit' vs 'reject' given the candidate and its 5 nearest library skills. Uncalibrated; memoised per prompt."""
    MODEL, REVISION = "Qwen/Qwen2.5-0.5B-Instruct", "7ae557604adf67be50417f59c2c2f167def9a775"

    def __init__(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(8)
        self.torch, self.cache = torch, {}
        self.tok = AutoTokenizer.from_pretrained(self.MODEL, revision=self.REVISION)
        self.m = AutoModelForCausalLM.from_pretrained(self.MODEL, revision=self.REVISION, torch_dtype=torch.float32).eval()
        self.ids = [self.tok.encode(w, add_special_tokens=False)[0] for w in ("admit", "reject")]

    def prompt(self, c, near):
        lib = "\n".join(f"- {s['name']}: {s['desc'][:300]}" for s in near)
        msg = ("You maintain an agent skill library. A router picks one skill for each task using only the skill name "
               f"and description.\n\nExisting skills most similar to the candidate:\n{lib}\n\nCandidate skill:\n"
               f"- {c['name']}: {c['desc'][:300]}\n\nIf the candidate overlaps or duplicates an existing skill so the router "
               "could confuse them, answer reject. If it is clearly distinct, answer admit. Answer with one word: admit or reject.")
        return self.tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)

    def p_admit(self, c, near):
        p = self.prompt(c, near)
        if p not in self.cache:
            with self.torch.no_grad():
                z = self.m(**self.tok(p, return_tensors="pt")).logits[0, -1, self.ids]
            self.cache[p] = float(self.torch.softmax(z, -1)[0])
        return self.cache[p]
