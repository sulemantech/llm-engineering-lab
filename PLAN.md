# Execution Plan

`docs/roadmap.md` explains **why and what**. This file covers **how and when**. Tick boxes as you go.
Every stage closes only when **all three** are true:

1. **Tests green:** `uv run pytest -m stageN` passes with no skips.
2. **Ship artifact exists:** it's committed, and the README results table is updated.
3. **Gate scored ≥ 7/8:** recorded in the scoreboard below, with no zero in any category.

Pace: 2–3 h/day, ~10 weeks. A buffer day is built into each week. If a day overruns by 50%, cut scope and log what you cut.

## Daily loop (copy into LOG.md)

| Block | Time | Output |
|---|---|---|
| Read | 20–30 min | 3–5 bullet notes in `paper-notes/` |
| Derive | 20–30 min | Shapes and equations on paper; photo or notes committed |
| Build | 60–90 min | Code plus tests; one commit per working step |
| Log | 10–15 min | `LOG.md` entry: did / broke / measured / next |

## Compute

| Where | Used for |
|---|---|
| This laptop (i7-1255U, 16 GB, CPU only) | All coding, tests, Stages 0–4, small runs, RAG with Ollama |
| Kaggle / Colab free GPU (T4) | Stage 5 bigger runs, Stage 7 LoRA/DPO, optional vLLM benchmark |

See `notebooks/gpu-runs.md`. Always record hardware in each result row.

---

## Week 0: Setup (2 days)
- [ ] `uv sync` works; `uv run pytest -m stage0` is green
- [ ] Push the repo to GitHub; CI is green
- [ ] Kaggle account set up; `torch.cuda.is_available()` is True in a Kaggle notebook
- [ ] Download TinyShakespeare into `data/` (see `data/README.md`)
- [ ] Gate: predict matmul output shapes before running them; explain gradient vs. parameter and optimizer vs. loss

**Ship:** green CI badge in the README.

## Week 1: Stage 1, LM fundamentals
- [ ] D1 `tokenizer/char.py`: CharTokenizer encode/decode
- [ ] D2 `data.py`: `get_batch` producing (x, y) with shifted targets
- [ ] D3 Compute cross-entropy for a 3-token vocab by hand, then verify with PyTorch
- [ ] D4 `bigram.py`: BigramLM with logits and loss, plus sampling
- [ ] D5 Training loop with a train/val split, seed, and a logged result row
- [ ] D6 *Break it on purpose:* wrong target shift and LR=10. Log the symptoms.
- [ ] D7 Buffer and gate

**Ship:** bigram val-loss row in `experiments/results/runs.jsonl` and in the README table.

## Week 2: Stage 2, attention
- [ ] D1 Read paper §1–3.2 and write `paper-notes/attention-is-all-you-need.md`
- [ ] D2 `attention/core.py`: `scaled_dot_product_attention`
- [ ] D3 `causal_mask`, padding mask, masked-position tests
- [ ] D4 `attention/mha.py`: MultiHeadAttention (split → attend → concatenate → project)
- [ ] D5 `attention/positional.py`: sinusoidal PE, plotted
- [ ] D6 Equation-to-code walkthrough in `paper-notes/equation-walkthrough.md`
- [ ] D7 Gate: **rewrite attention from a blank file**, then run the tests against it

**Ship:** `pytest -m stage2` is green, including parity with `F.scaled_dot_product_attention`.

## Week 3: Stage 3, original Transformer
- [ ] FFN, residual + LayerNorm (post-norm as in the paper; note pre-norm)
- [ ] EncoderLayer → Encoder stack with embeddings and PE
- [ ] DecoderLayer (masked self-attention + cross-attention) → Decoder
- [ ] Full model, target shifting, sequence loss
- [ ] Train on a synthetic reverse/copy task until near 100% accuracy
- [ ] Architecture diagram in `architecture/` (Mermaid is fine)
- [ ] Gate tests: shapes for changed layer/head counts; future targets masked; decoder uses the encoder

**Ship:** diagram plus synthetic-task accuracy row.

## Week 4: Stage 4, GPT from scratch
- [ ] GPTConfig, Block, GPT (embeddings → blocks → LN → lm_head, weight tying)
- [ ] Train on TinyShakespeare (CPU: ~4 layers, 4 heads, 128–256 dim, context 128)
- [ ] Checkpoint save/load/resume
- [ ] Generation: greedy, temperature, top-k, top-p
- [ ] **Gate test:** load the GPT-2 124M weights into *your* GPT and match HF logits within 1e-4
- [ ] **Gate test:** leakage test (changing token t+1 leaves logits ≤ t unchanged)
- [ ] Introduce a mask bug, find it with the tests, write it up in LOG.md

**Ship:** sample generations plus the GPT-2 parity test. **This is your strongest portfolio proof.**

## Week 5: Stage 5a, modern GPT + BPE
- [ ] Byte-level BPE: train, encode, decode, round-trip tests on random Unicode
- [ ] One change per run, logged as an ablation: RMSNorm, RoPE, SwiGLU, GQA, AdamW+wd, warmup+cosine, grad clipping
- [ ] TinyStories run on a Kaggle GPU with bf16

**Ship:** ablation table (config → val loss → tokens/s → hardware).

## Week 6: Stage 5b, experiments and scaling
- [ ] Parameter count by formula vs. `sum(p.numel())`
- [ ] FLOPs ≈ 6·N·D; memory estimate vs. measured; state what the estimate leaves out
- [ ] Profile one training step (`torch.profiler`)
- [ ] Re-run one config with the same seed and report the variance
- [ ] `experiments/report.md` with limitations

**Ship:** experiment report.

## Week 7: Stage 6a, adapt and evaluate (Kaggle GPU)
- [ ] Choose a use case; write the "prompting vs RAG vs fine-tuning" justification
- [ ] Golden eval set (50–100 items) frozen *before* tuning, with failure categories
- [ ] Baseline: small instruct model (SmolLM/Qwen ~0.5–1.5B)
- [ ] LoRA/QLoRA SFT, then evaluate before and after
- [ ] LLM-as-judge plus a check of the judge against your own labels on 20 items
- [ ] Optional: small DPO demo

**Ship:** before/after table including failure cases.

## Week 8: Stage 6b, inference and serving
- [ ] Your KV cache: same greedy tokens, speedup measured
- [ ] Latency, TTFT and throughput vs. batch size (documented conditions)
- [ ] Quantization: int8/4-bit or GGUF via llama.cpp/Ollama, comparing memory, speed and quality
- [ ] FastAPI service: validation, errors, logging, model version, tests, Dockerfile
- [ ] Optional: vLLM comparison on Kaggle

**Ship:** benchmark table plus a running API.

## Week 9: RAG prototype
- [ ] Ingest → chunk → embed → vector store (FAISS/Chroma)
- [ ] Hybrid retrieval (BM25 + vectors) and reranking
- [ ] Cited answers via Ollama or an API
- [ ] Eval: recall@k, MRR, faithfulness, and "I don't know" behavior
- [ ] Per-user document access filter, plus prompt-injection test documents

**Ship:** RAG eval table plus a demo.

## Week 10: Architect capstone
- [ ] Requirements and measurable success criteria
- [ ] Decision matrix: hosted vs. open, build vs. buy (ADR)
- [ ] Deployment diagram and request flow; capacity/cost estimates with labelled assumptions
- [ ] Eval and release gates; observability and alerting; risk register
- [ ] Two designs compared; selected design defended
- [ ] README polish, a 5-minute demo video, and a write-up

**Ship:** portfolio-ready repo.

---

## Gate scoreboard

0 = not demonstrated · 1 = partial/needed hints · 2 = independent and verified. Pass: ≥ 7/8, no zeros.
Attention math, causal masking, target alignment and eval integrity require a 2.

| Stage | Concept | Implement | Evidence | Debug/trade-offs | Total | Date |
|---|---|---|---|---|---|---|
| 0 | | | | | | |
| 1 | | | | | | |
| 2 | | | | | | |
| 3 | | | | | | |
| 4 | | | | | | |
| 5 | | | | | | |
| 6 | | | | | | |
| 7 | | | | | | |
