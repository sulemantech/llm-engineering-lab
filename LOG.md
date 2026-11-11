# Daily Log

Newest entry on top. Five minutes per day. Be honest: failures are the most useful entries.

<!--
## YYYY-MM-DD · Week N Day N · <topic>
- **Did:**
- **Broke / confused:**
- **Fixed by:**
- **Measured:** (run id, metric, hardware)
- **Next:**
- **Time:** h
-->

## 2026-11-11 · Week 1 catch-up · baseline + housekeeping
- **Did:** Timed the CPU baseline. Ollama is installed (date not recorded). Ticked the finished items in PLAN.md. Wrote the earlier entries below from the git history.
- **Broke / confused:** No log entries for 8 days.
- **Fixed by:** Added them now from the commits. From here on, log the same day.
- **Measured:** 1024×1024 fp32 matmul = **17.3 ms** (~124 GFLOP/s). Mean of 50 runs after 5 warm-up runs, torch 2.15.0+cpu, 10 threads, i7-1255U. This is the reference speed.
- **Next:** Retake RNN self-check Q4/Q6. Write the D3 embeddings explanation. Then D4 `BigramLM`.
- **Time:** h

## 2026-11-03 · Week 1 Day 1–3 · tokenizer, batches, LM math
- **Did:** Read about RNN/LSTM limits; self-check scored 7/12 (`paper-notes/attention-is-all-you-need.md`). Wrote `CharTokenizer` and `get_batch` (next-token targets). Worked dot products, matmul, softmax and cross-entropy by hand (photos in `architecture/notes/`), then checked them in `scratch/day3.py`, along with `one_hot @ W == nn.Embedding`.
- **Broke / confused:** Q4 vanishing gradients (0/2). Q6: I listed parallelism as a cost; it's actually a benefit.
- **Fixed by:** Wrote refined answers in the notes. Retake still to do.
- **Measured:** `pytest -m stage1`: 4 passed (tokenizer + batch), 3 skipped (bigram).
- **Next:** BigramLM.
- **Time:** h

## 2026-11-02 · Week 0 · plan and compute
- **Did:** Added psutil, the Transformer-training week, the Kaggle compute plan, and the Kaggle workflow doc (`notebooks/gpu-runs.md`).
- **Next:** Week 1.
- **Time:** h

## 2026-10-30 · Week 0 · scaffold
- **Did:** Set up the repo with uv, the gate tests for each stage, and the roadmap.
- **Time:** h
