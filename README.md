# LLM Engineering Lab

From *Attention Is All You Need* to a tested GPT, an adapted model, a served API, and a defended
enterprise LLM architecture, built from scratch and measured honestly.

- **Plan / progress:** [PLAN.md](PLAN.md) · **Daily log:** [LOG.md](LOG.md) · **Roadmap:** [docs/roadmap.md](docs/roadmap.md)

## Quickstart

```bash
uv sync                      # CPU PyTorch + dev tools
uv run pytest                # all gate tests (unimplemented work shows as SKIPPED)
uv run pytest -m stage2      # one stage's gate tests
uv run ruff check .
```

Test status works as a to-do list: **skipped** = not implemented yet, **failed** = implemented but wrong,
**passed** = done.

## Layout

| Path | Purpose |
|---|---|
| `src/llmlab/` | Library code: tokenizer, attention, transformer, gpt, training, generation, serving, rag, evaluation |
| `tests/` | Gate tests, one file per stage |
| `experiments/configs/` | One YAML per run |
| `experiments/results/runs.jsonl` | One line per run (appended by `llmlab.utils.log_result`) |
| `paper-notes/` | Paper notes and equation-to-code walkthroughs |
| `architecture/` | Diagrams, ADRs, capacity estimates, risk register |
| `notebooks/` | Kaggle/Colab GPU runners |
| `data/` | Datasets (git-ignored; see `data/README.md`) |

## Results

| Stage | Model / config | Data | Metric | Value | Hardware | Run id |
|---|---|---|---|---|---|---|
| | | | | | | |
