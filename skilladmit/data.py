"""Real public skill metadata -> a pool of skills with routing queries derived from their own SKILL.md bodies.

Every skill is a real SKILL.md from a pinned public repo. Router input is name + description (frontmatter).
Routing queries are the bullets of the skill's own "When to use" section (author-written task statements);
the gold label of a query is the skill it came from. Labels are DERIVED, not annotated: see docs/DATA.md.
SYNTH near-duplicates are string edits of a real skill, flagged synth=True, and have no queries of their own.
"""
import gzip, json, re, subprocess, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "src"
POOL = ROOT / "data" / "skills.jsonl.gz"
SOURCES = {  # repo -> (commit, repo-level license or None if per-skill LICENSE.txt decides)
    "anthropics/skills": ("ca45fc496a3436251f191354091d557da31c15b5", None),
    "openai/skills": ("49f948faa9258a0c61caceaf225e179651397431", None),
    "huggingface/skills": ("25910b3d0ba31a2ba102f901cdc9e008c278a69c", "Apache-2.0"),
    "K-Dense-AI/claude-scientific-skills": ("a25cf9ec00e2745d03f8704acf6f6919694ccd9a", "MIT"),
    "obra/superpowers": ("2ce8aa15d05bd57213e6f746f953aafb7ded7a8b", "MIT"),
    "wshobson/agents": ("7e2fb559cf4b824b27f1fec49fe88f7511066d26", "MIT"),
    "alirezarezvani/claude-skills": ("c85b17e1fb621ba760478baaacc4fcb9be225e74", "MIT"),
    "github/awesome-copilot": ("8178d1939a203f691067073dcab2a3f1e14072d3", "MIT"),
    "microsoft/skills": ("db273d02235f048331ca1597c0a98d529da1cabc", "MIT"),
}
WHEN = re.compile(r"^#{2,4}\s*(when to (use|invoke|apply)|use (this skill )?when|use cases|triggers?|trigger phrases)\b",
                  re.I)
STOP = re.compile(r"^\s*(\*\*|__)?(do not|don't|not for|avoid|skip|when not)", re.I)
MIN_Q, MIN_W, MAX_W = 4, 4, 40


def fetch():
    for repo, (sha, _) in SOURCES.items():
        d = SRC / repo.replace("/", "__")
        if not (d / ".git").exists():
            d.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", "-q"], cwd=d, check=True)
            subprocess.run(["git", "remote", "add", "origin", f"https://github.com/{repo}"], cwd=d, check=True)
        if subprocess.run(["git", "rev-parse", "HEAD"], cwd=d, capture_output=True, text=True).stdout.strip() != sha:
            subprocess.run(["git", "fetch", "-q", "--depth", "1", "origin", sha], cwd=d, check=True)
            subprocess.run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=d, check=True)


def skill_license(path, repo_lic):
    for f in path.parent.iterdir():
        if f.name.lower().startswith("license"):
            t = f.read_text(errors="replace")[:600]
            return "Apache-2.0" if "Apache License" in t else "MIT" if "Permission is hereby granted" in t else None
    return repo_lic


def clean(s):
    s = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", s)
    s = re.sub(r"[*_`>#]|<[^>]+>", "", s)
    return re.sub(r"\s+", " ", s).strip(" -:;.\"'")


def queries(body):
    out, on = [], False
    for line in body.splitlines():
        if re.match(r"^#{1,6}\s", line):
            on = bool(WHEN.match(line)) and "not" not in line.lower()
            continue
        if not on:
            continue
        if STOP.match(line.lstrip("-* ")):
            on = False
            continue
        m = re.match(r"^\s{0,3}(?:[-*+]|\d+[.)])\s+(.*)", line)
        if m:
            q = clean(m.group(1))
            if MIN_W <= len(q.split()) <= MAX_W:
                out.append(q)
    return out


def parse(path):
    text = path.read_text(errors="replace")
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n(.*)$", text, re.S)
    if not m:
        return None
    try:
        fm = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return None
    if not isinstance(fm, dict) or not isinstance(fm.get("name"), str) or not isinstance(fm.get("description"), str):
        return None
    return fm["name"].strip(), clean(fm["description"]), queries(m.group(2))


def build():
    fetch()
    seen_name, seen_desc, seen_q, pool = set(), set(), {}, []
    for repo, (sha, repo_lic) in SOURCES.items():
        d = SRC / repo.replace("/", "__")
        for p in sorted(d.rglob("SKILL.md")):
            lic = skill_license(p, repo_lic)
            r = lic and parse(p)
            if not r:
                continue
            name, desc, qs = r
            key = re.sub(r"[^a-z0-9]", "", name.lower())
            if key in seen_name or desc.lower() in seen_desc or len(desc.split()) < 5:
                continue  # first source wins on a repeated name or identical description (cross-repo copies)
            seen_name.add(key); seen_desc.add(desc.lower())
            qs = [q for q in dict.fromkeys(qs) if q.lower() not in desc.lower()]
            for q in qs:
                seen_q.setdefault(q.lower(), set()).add(key)
            pool.append(dict(id=f"{repo}:{p.parent.relative_to(d)}", name=name, desc=desc, queries=qs,
                             repo=repo, commit=sha, path=str(p.relative_to(d)), license=lic, synth=False))
    for s in pool:  # a query shared by two skills has no single gold skill: drop it everywhere
        s["queries"] = [q for q in s["queries"] if len(seen_q[q.lower()]) == 1]
    pool = [s for s in pool if len(s["queries"]) >= MIN_Q]
    POOL.parent.mkdir(exist_ok=True)
    with gzip.open(POOL, "wt") as f:
        for s in pool:
            f.write(json.dumps(s) + "\n")
    return pool


def load():
    with gzip.open(POOL, "rt") as f:
        return [json.loads(l) for l in f]


SUFFIX = ["helper", "tools", "assistant", "pro", "kit", "toolkit", "expert", "workflow"]


def synth_dup(s, rng, drop=0.2):
    """SYNTH near-duplicate of a real skill: renamed (name-suffix) and 20% of description words dropped at random."""
    words = s["desc"].split()
    keep = [w for w in words if rng.random() >= drop] or words
    return dict(id=f"SYNTH:{s['id']}:{rng.integers(1 << 30)}", name=f"{s['name']}-{SUFFIX[rng.integers(len(SUFFIX))]}",
                desc=" ".join(keep), queries=[], synth=True, dup_of=s["id"], repo=s["repo"], license=s["license"])


def synth_blend(a, b, rng):
    """SYNTH partial overlap: b's name with a suffix, description = a's first sentence + the rest of b's description."""
    sa, sb = (re.split(r"(?<=[.!?])\s+", x["desc"]) for x in (a, b))
    rest = sb[1:] or sb
    return dict(id=f"SYNTH:{a['id']}+{b['id']}:{rng.integers(1 << 30)}", name=f"{b['name']}-{SUFFIX[rng.integers(len(SUFFIX))]}",
                desc=" ".join([sa[0]] + rest), queries=[], synth=True, dup_of=a["id"], repo=a["repo"], license=a["license"])


def episode(pool, rng, k0=40, n_new=12, n_copy=6, n_blend=6):
    """One lifelong stream: k0 base skills, then n_new real unseen skills + SYNTH copies/blends of base skills, shuffled."""
    idx = rng.permutation(len(pool))
    base, new, rest = [pool[i] for i in idx[:k0]], [pool[i] for i in idx[k0:k0 + n_new]], [pool[i] for i in idx[k0 + n_new:]]
    cands = new + [synth_dup(base[j], rng) for j in rng.choice(k0, n_copy, replace=False)]
    cands += [synth_blend(base[j], rest[m], rng) for j, m in zip(rng.choice(k0, n_blend, replace=False),
                                                                  rng.choice(len(rest), n_blend, replace=False))]
    return base, [cands[i] for i in rng.permutation(len(cands))]


if __name__ == "__main__":
    pool = build() if "--cached" not in sys.argv else load()
    from collections import Counter
    nq = [len(s["queries"]) for s in pool]
    print(len(pool), "skills,", sum(nq), "queries, median", sorted(nq)[len(nq) // 2], "per skill")
    print(Counter(s["repo"] for s in pool).most_common())
    print(Counter(s["license"] for s in pool).most_common())
