# CLAUDE.md — Radiance (Bible Worlds · Unified 3DGS Asset PoC)

> Keep this lean: always-on project conventions only. The step-by-step pipeline lives in `docs/head_mp4_to_3dgs_pipeline.org`.

## Project identity
Codename: **Radiance**. Part of the **Bible Worlds** program — historically and scripturally accurate 3D world environments for media production (MediaCenter-affiliated).
Current work: a PoC proving that a **single canonical 3DGS asset (Peter) supports relight, edit, and compose**.

## Canonical principles (hard rules)
- **OpenUSD is the canonical file format** (file-based, not a database). No new server infrastructure needed; existing MAM storage suffices.
- **splat=background / mesh=foreground is a governance default, not an absolute rule.** Named biblical figures default to mesh-based Digital Persons, but foreground objects and named figures may be 3DGS in appropriate contexts.
- **MAM handles cross-scene asset discovery; USD handles intra-scene querying.**
- **AI renders/processes but does not define canonical world content.** The asset is canonical (a superset of USD primvars); methods are swappable.

## Current PoC decisions (do not relitigate)
- primitive = **3DGS** (not 2DGS): homogeneity with the team / ecosystem / USD schema / compose targets, plus volumetric robustness under free camera movement. The 2DGS normal advantage is mitigated with normal priors.
- A **single asset** carries relight + edit + compose: {geometry + normal} + {PBR material} + {identity}, anchored in USD (explicit transform + metric scale).
- base = **Relightable 3D Gaussians**; identity = **Gaussian Grouping** (both 3DGS-native).
- Framing for relight / edit / compose: not "intrinsically impossible" but **"still difficult in production, and especially difficult to retrofit onto an existing splat."**
- License: research code (INRIA-rasterizer-based) is **PoC / internal demo only**. Reimplement on **gsplat (Apache 2.0)** for production.

## Working conventions
- **Respond in Korean for technical sessions**; keep technical terms in English (OpenUSD, 3DGS, etc.).
- English communications for international colleagues: **short and natural** (no padding or over-elaboration). Use "We" framing for team communications.
- **Build first, optimize later** (e.g., assess decomposition quality before designing preprocessing).
- Evaluate tools against explicit criteria: license (commercial usability), hardware compatibility, export fidelity, governance alignment.
- Keep infrastructure separated: interactive / visual work = **Xesktop**, batch compute = **RunPod**.

## Execution environment / Skills
- Execution topology: **Claude Code and Emacs run locally; code executes on the remote Pod kernel via emacs-jupyter.**
- **For REPL / org-babel / org-file work, use the `org-babel-repl` skill** (runs jupyter-python on the remote kernel via the `eval_elisp` MCP `darren-org-*` functions).
- Use the terminal for standard script execution, builds, tests, and git (not a skill).
