# ICL Structured Memory

In-context learning system extended with four named memory sections that
persist across questions within a run and survive context overflow.

## How It Works

The system maintains a structured memory of four sections:

| Section | What It Stores |
|---|---|
| `schema_map` | Table names, columns, join keys, data types the agent discovered |
| `format_quirks` | Non-obvious encodings (price units, timestamp formats, boolean encoding) |
| `drift_log` | Schema changes, renamed tables, stale columns, SQL errors after drift |
| `verified_joins` | Join patterns that produced correct answers |

### Memory Lifecycle

1. **Empty at run start** — Q1 starts with no memory injection (avoids overhead with empty sections).
2. **Populated from feedback** — the `observe()` hook parses harness feedback after each question and appends facts to the relevant section.
3. **Optional model writes** — the response schema includes four `*_add` fields. The model may write a single short fact to any section as part of its structured response.
4. **Injected at instance boundary** — when a new question starts and memory has content, the full block is injected as a `user` message before the question.
5. **Drift handling** — when the harness signals migration, `schema_map` and `format_quirks` entries are prefixed with `[UNVERIFIED]` to force the agent to re-verify before reuse.

### Response Schema Augmentation

The task's response schema is wrapped with four optional fields:

```
schema_map_add:      Optional[str]
format_quirks_add:   Optional[str]
drift_log_add:       Optional[str]
verified_joins_add:  Optional[str]
```

Following the same pattern as `icl_notepad` — the model can append memory
entries inline with its action response.

## Parameters

- `model` (default `gpt-5.4`)
- `max_tokens` (default: inferred from model)
- `name` (default `icl_structured_memory`)
- `reserve_tokens` (default 500)
- `provider_mode` (default `auto`)

## Results

Tested on `database_exploration` with default schedule (5 runs, permute):

| Metric | Value |
|---|---|
| Mean Cumulative Reward | 16.933 |
| Mean Cumulative Gain | +11.533 |
| Mean Accuracy | 49% |
| Avg Queries per Question | 2.05 |

This places the system above `icl-gpt-5.4` (13.880) and just below
`mem0-gpt-5.4` (17.240) on the database_exploration leaderboard.

## Running

```bash
clbench run database_exploration \
  --system icl_structured_memory \
  --task-params '{"schedule": "default"}' \
  --system-params '{"model": "gpt-5.4"}'
```

## Files

- `system.py` — system implementation
- `__init__.py` — package registration
