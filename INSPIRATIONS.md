# Inspirations

- **Skill++** ([arXiv:2611.00544](https://arxiv.org/abs/2611.00544)): shows that under progressive disclosure a new,
  similar skill can displace an earlier one even when the earlier skill and the router are unchanged. skilladmit
  takes that failure as the thing to bound, and leaves the router alone.
- **Learn then Test** (Angelopoulos, Bates, Candès, Jordan, Lei, [arXiv:2110.01052](https://arxiv.org/abs/2110.01052)):
  risk control by multiple testing over a configuration grid. `skilladmit/ltt.py` is Hoeffding-Bentkus + Bonferroni.
- **Decision Models as Policy Gates for Coding Agents** (Beno, [arXiv:2611.02155](https://arxiv.org/abs/2611.02155))
  and **sarvamai/prism** ([model card](https://huggingface.co/sarvamai/prism)): typed decision models over a fixed
  option set as fast gates. Prism itself needs ~120 GB of GPU memory, so it is not run here.
- **TalentRL** ([arXiv:2610.41143](https://arxiv.org/abs/2610.41143)), **Structure-Aware Skill Retrieval**
  ([arXiv:2610.41909](https://arxiv.org/abs/2610.41909)), **SEEK** ([arXiv:2609.29803](https://arxiv.org/abs/2609.29803))
  and **Skill-RAG** ([arXiv:2604.15771](https://arxiv.org/abs/2604.15771)): learning which skill to pick. skilladmit
  is about which skills get to exist.
- **prefixgate** ([Sebastianrodaaa/prefixgate](https://github.com/Sebastianrodaaa/prefixgate)): the earlier streaming
  LTT gate on tool-call prefixes; the LTT code and dashboard skeleton come from there.
- The public skill libraries in `NOTICE.md`, whose authors wrote the "When to use" lists that became the task set.
- **athanor** ([ghgoodreau/athanor](https://github.com/ghgoodreau/athanor)): the README shape.
