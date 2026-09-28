# v2/clm — CLM-8B System One scoring (local)

CLM is a "System One" model: it scores candidate actions against a state
instead of generating text. Given a state and a closed option set, it returns
a probability distribution. This package wires a **local CLM-8B service**
(`clm-serve`; CLM-v0.1-8B = frozen Qwen3-8B encoder + trained projection
heads) into the v2 pipeline as an alpha model, paired with the Kelly sizing
policy in `v2/portfolio/kelly.py`.

## Layout

| Piece | What |
|---|---|
| `client.py` | HTTP client for a local clm-serve (`POST /v1/systemone`); raises `CLMError` on any failure |
| `cache.py` | One JSON per scoring decision (`.v2_cache/clm/`), mirroring `v2/llm/cache.py` |
| `../signals/clm.py` | `CLMAnalyst`: snapshot → up/flat/down probabilities → `Signal` |
| `../portfolio/kelly.py` | `blend_kelly`: probabilities → buffered Kelly weights; `kelly_fraction` = f* = p − (1−p)/b |

## Prerequisites

The scoring service runs outside this repo (default `http://127.0.0.1:8700`):

```bash
~/clm/scripts/start-clm.sh      # embeddings (:8092) + clm-serve (:8700)
```

Env overrides: `CLM_BASE_URL` (service URL), `CLM_MODEL` (default `clm-latest`).

## Use

Strategy YAML:

```yaml
models:
  - name: clm
blend:
  method: kelly
  payoff_ratio: 1.0
  buffer: 0.2          # 20% of full Kelly discarded — the safety margin
  gross_target: 1.0    # scale-down-only cap on gross exposure
```

One cycle:

```bash
poetry run python -m v2.run v2/funds/clm-kelly.yaml --date 2025-06-03
```

Compare sizing policies on the same signals:

```bash
poetry run python -m v2.run v2/funds/clm-conviction.yaml --date 2025-06-03
```

## Kelly sizing, precisely

- p (win probability) is taken **conditional on the quarter resolving**; a
  flat outcome is a push that contributes 0 to expected log wealth and drops
  out: `p = p_up / (p_up + p_down)`, `q = p_down / (p_up + p_down)`.
- `f* = p − (1 − p) / b`; **no edge (f* ≤ 0) → no position.**
- The **buffer** reserves a fraction of the budget: the book deploys
  `min(Σf*, gross_target) × (1 − buffer)`, split across names in proportion
  to f*. Under the cap that is exactly the per-bet fractional-Kelly stake
  `f* × (1 − buffer)`; at the cap it holds Kelly proportions on the buffered
  budget. Either way full Kelly never deploys more than 80% of the book
  (default buffer 0.2), and the rest is cash.
- The fund's risk limits remain the master gate afterward
  (max_position_pct / max_gross_exposure clamps).
- Signals without `metadata.probabilities` fall back to
  `p = (1 + |conviction|) / 2` so any model in the registry can feed the blend.

## Failure contract

- Data errors propagate (fail loud).
- CLM call/parse failures abstain (`Signal(value=0, metadata.abstained)`):
  one dead local service must not crash a backtest.
- An unchanged snapshot never pays a second scoring call (disk cache).

## Tests

```bash
poetry run pytest v2/clm v2/signals/test_clm_analyst.py \
    v2/portfolio/test_kelly.py v2/pipeline/test_run_cycle_kelly.py -q
```

Integration guide (HTML, with live-run numbers):
`docs/clm-kelly-guide.html`.
