# Execution Plan

`docs/roadmap.md` explains **why and what**. This file covers **how and when**. Tick boxes as you go.
Every stage closes only when **all three** are true:

1. **Tests green:** `uv run pytest -m stageN` passes with no skips.
2. **Ship artifact exists:** it's committed, and the README results table is updated.
3. **Gate scored ≥ 7/8:** recorded in the scoreboard below, with no zero in any category.

Pace: 2–3 h/day, ~11 weeks. A buffer day is built into each week. If a day overruns by 50%, cut scope and log what you cut.

## Daily loop (copy into LOG.md)

| Block | Time | Output |
|---|---|---|
| Read | 20–30 min | 3–5 bullet notes in `paper-notes/` |
| Derive | 20–30 min | Shapes and equations on paper; photo or notes committed |
| Build | 60–90 min | Code plus tests; one commit per working step |
| Log | 10–15 min | `LOG.md` entry: did / broke / measured / next |

## Compute: laptop for building, Kaggle for heavy runs

| Where | Used for |
|---|---|
| **Laptop** (i7-1255U, 16 GB, CPU only) | All coding, tests, debugging, small runs (Weeks 0–5), Ollama/RAG (Weeks 9–10) |
| **Kaggle** (free T4 16 GB, ~30 GPU h/week) | Heavy runs: Week 4 translation, Week 6 TinyStories, Week 7 GPU measurements, Week 8 LoRA/QLoRA, Week 9 vLLM |

How it works (see `notebooks/gpu-runs.md`): code on the laptop → tests green → `git push` → a Kaggle notebook pulls the repo and runs a config → download `runs.jsonl` → commit the results.

**Rules**
1. **Never debug on Kaggle.** A script goes to Kaggle only after it runs for 50 steps on CPU with a tiny config. That saves GPU hours.
2. **Same code everywhere:** `device = llmlab.utils.get_device()`; no hard-coded `"cuda"`.
3. **T4 = fp16**, not bf16: use `torch.autocast("cuda", dtype=torch.float16)` with `torch.amp.GradScaler`.
4. **Checkpoint every ~10 min to `/kaggle/working`** and support `--resume`. Sessions can end.
5. **Long runs use "Save & Run All (Commit)"**, so they keep running after you close the browser.
6. **Budget:** plan ≤ 20 GPU h/week and keep 10 h spare. Log the GPU hours used in LOG.md.
7. **CPU fallback:** every Kaggle task has a smaller CPU version (see the table below). If the quota or internet fails, run that and log it.
8. **Compare at equal settings:** same steps, data and tokenizer. Label every number with its hardware (CPU or T4); `log_result` does this automatically.
9. On CPU: `torch.set_num_threads(10)`, laptop plugged in, heavy apps closed.

**Size budget (CPU fallback / laptop runs)**

| Week | Model | Data | Expected run time |
|---|---|---|---|
| 1 | Bigram / MLP | TinyShakespeare (1 MB) | seconds–minutes |
| 3–4 | Encoder–decoder, 2+2 layers, d_model 128, 4 heads | copy task; Multi30k subset (≤ 10k pairs, ≤ 20 tokens) | 5 min; ~1 h |
| 5 | GPT, 4 layers, 4 heads, d_model 128–256, context 128 | TinyShakespeare | 10–40 min |
| 5 | GPT-2 124M (inference only, parity test) | n/a | seconds |
| 6 | GPT ~1–5M params, BPE vocab 2–4k | TinyStories subset (20–50 MB) | 15–30 min per ablation, one overnight run |
| 8 | SmolLM2-135M (360M max), LoRA in fp32 | 500–1,000 short examples, seq ≤ 256 | 30–90 min |
| 9–10 | Ollama Q4 models, 0.5B–3B (e.g. qwen2.5:1.5b, llama3.2:3b) | n/a | ~5–20 tokens/s |

**Size budget (Kaggle T4)**

| Week | Model | Data | Expected GPU time |
|---|---|---|---|
| 4 | Encoder–decoder, 3+3 layers, d_model 256 | Full Multi30k (~29k pairs) | ~1 h |
| 6 | GPT ~10–30M params, BPE vocab 4–8k, context 256 | TinyStories (~0.5–1 GB) | 2–4 h |
| 7 | Same configs as the CPU runs | — | ~1 h (CPU vs GPU comparison) |
| 8 | Qwen2.5-0.5B/1.5B-Instruct, LoRA fp16 and QLoRA 4-bit | 2–5k examples, seq 512 | 1–3 h |
| 9 | vLLM serving a 0.5B–1.5B model | — | ~1 h |

---

## Week 0: Setup (2 days)
- [ ] `uv sync` works; `uv run pytest -m stage0` is green
- [ ] Push the repo to GitHub; CI is green
- [ ] Measure the CPU baseline: time a 1024×1024 matmul and record it in LOG.md as your reference speed
- [ ] Install Ollama for Windows (needed in Weeks 9–10); `ollama run qwen2.5:0.5b` works
- [ ] Kaggle account with **phone verified** (needed for GPU and internet); in a GPU T4 notebook, `torch.cuda.is_available()` is True
- [ ] Smoke test: the Kaggle notebook clones this repo, installs it, and runs `pytest -m stage0` on the GPU
- [ ] Download TinyShakespeare into `data/` (see `data/README.md`)
- [ ] Gate: predict matmul output shapes before running them; explain gradient vs. parameter and optimizer vs. loss

**Ship:** green CI badge in the README.

## Week 1: Stage 1, LM fundamentals
- [ ] D1 Read: what problem Transformers solved. RNN/LSTM process tokens one step at a time, so there is no parallelism over the sequence, long-range information must pass through every step, and gradients vanish. Write notes in `paper-notes/attention-is-all-you-need.md` § "Problem".
- [ ] D1 `tokenizer/char.py`: CharTokenizer encode/decode
- [ ] D2 `data.py`: `get_batch` producing (x, y) with shifted targets
- [ ] D3 By hand, then verified in PyTorch: dot products, a matmul as many dot products, softmax, and cross-entropy for a 3-token vocab
- [ ] D3 Embeddings: show that `one_hot(idx) @ W == nn.Embedding(idx)`. Explain why learned dense vectors beat raw ids or one-hot.
- [ ] D4 `bigram.py`: BigramLM with logits and loss, plus sampling
- [ ] D5 Training loop with a train/val split, seed, and a logged result row
- [ ] D6 *Break it on purpose:* wrong target shift and LR=10. Log the symptoms.
- [ ] D7 Buffer and gate: **explain how a sentence becomes a sequence of vectors** (text → tokens → ids → embedding rows) and why embeddings are needed

**Ship:** bigram val-loss row in `experiments/results/runs.jsonl` and in the README table.

## Week 2: Stage 2, attention
- [ ] D1 Read paper §1–3.2 and write `paper-notes/attention-is-all-you-need.md`
- [ ] D2 `attention/core.py`: `scaled_dot_product_attention`
- [ ] D2 **Why divide by √d_k:** for random q, k, measure Var(q·k) for d_k = 4, 64, 512 (≈ d_k). Plot softmax with and without scaling, and show that unscaled scores saturate to near one-hot with tiny gradients.
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
- [ ] Sanity check: overfit one tiny batch
- [ ] Architecture diagram in `architecture/` (Mermaid is fine), **annotated with tensor shapes at every component**
- [ ] Gate tests: shapes for changed layer/head counts; future targets masked; decoder uses the encoder
- [ ] Gate: draw the full architecture from memory and walk one batch's shapes through every component

**Ship:** shape-annotated diagram plus a green forward-pass test suite.

## Week 4: Stage 3b, training the original Transformer
- [ ] Forward → loss → backward on one example, tracing which tensors hold gradients
- [ ] Adam: implement one Adam step by hand (m, v, bias correction) and match `torch.optim.Adam`. Explain why it beats plain SGD here.
- [ ] Paper LR schedule: warmup, then inverse square-root decay (`d_model^-0.5 · min(step^-0.5, step·warmup^-1.5)`). Plot it and explain why warmup is needed.
- [ ] Label smoothing (ε = 0.1): implement it yourself, compare with `F.cross_entropy(label_smoothing=0.1)`, and explain why the loss floor rises while accuracy can improve
- [ ] Variable-length batching: padding, source padding mask, `ignore_index` for pad tokens in the loss
- [ ] Greedy decoding for the encoder–decoder (beam search optional)
- [ ] **Copy task:** train until ~100% accuracy. Plot the loss curve and explain each phase (start ≈ ln V, plateau, drop, floor).
- [ ] **Translation:** Multi30k de→en with a pretrained subword tokenizer for now (you build BPE in Week 6). Report BLEU (sacrebleu).
  - First, a CPU check: a 1k-pair subset for 50 steps, loss going down
  - **Kaggle:** full Multi30k, 3+3 layers, d_model 256, fp16, ~1 h
  - *CPU fallback:* ≤ 10k pairs of ≤ 20 tokens, 2+2 layers, d_model 128, ~1 h; expect a lower BLEU and explain why
- [ ] **Attention visualization:** heatmaps of encoder self-attention, decoder masked self-attention, and cross-attention (which looks like word alignment)
- [ ] Gate: demonstrate the working model, explain its limitations, and defend the main design choices (post-norm, sinusoidal PE, shared embeddings, label smoothing, warmup)

**Ship:** copy-task loss curve, BLEU row, attention heatmaps.

## Week 5: Stage 4, GPT from scratch
- [ ] GPTConfig, Block, GPT (embeddings → blocks → LN → lm_head, weight tying)
- [ ] Train on TinyShakespeare (~4 layers, 4 heads, 128–256 dim, context 128; aim for val loss ≈ 1.5–1.7 on chars)
- [ ] Checkpoint save/load/resume
- [ ] Generation: greedy, temperature, top-k, top-p
- [ ] **Gate test:** load the GPT-2 124M weights into *your* GPT and match HF logits within 1e-4
- [ ] **Gate test:** leakage test (changing token t+1 leaves logits ≤ t unchanged)
- [ ] Introduce a mask bug, find it with the tests, write it up in LOG.md

**Ship:** sample generations plus the GPT-2 parity test. **This is your strongest portfolio proof.**

## Week 6: Stage 5a, modern GPT + BPE
- [ ] Byte-level BPE: train, encode, decode, round-trip tests on random Unicode
- [ ] Train BPE on a TinyStories subset (vocab 4–8k) on the laptop; save the tokenizer file and commit it
- [ ] Ablations on the **laptop** at equal steps (15–30 min each): RMSNorm, RoPE, SwiGLU, GQA, AdamW+wd, warmup+cosine, grad clipping. Small models are enough to see the relative differences, and this saves GPU hours.
- [ ] **Kaggle:** the best config at ~10–30M params on full TinyStories, fp16, `F.scaled_dot_product_attention` (uses fused kernels on GPU), 2–4 h committed run with checkpoints. Should produce coherent short stories.
- [ ] **Kaggle:** fp32 vs fp16 autocast, and with vs without `torch.compile`: tokens/s and memory for 200 steps each
- [ ] Notes: why the T4 uses fp16 rather than bf16, and what loss scaling does
- [ ] *CPU fallback:* overnight run of the best config on a 20–50 MB subset

**Ship:** ablation table (config → val loss → tokens/s → hardware).

## Week 7: Stage 5b, experiments and scaling
- [ ] Parameter count by formula vs. `sum(p.numel())`
- [ ] FLOPs ≈ 6·N·D; memory estimate vs. measured (process RSS via `psutil`); state what the estimate leaves out
- [ ] Measure tokens/s on CPU *and* on the Kaggle T4 for the same config. Compare with the FLOPs-based estimate and explain the gap (memory bandwidth, kernel overhead, utilization).
- [ ] GPU memory on Kaggle: `torch.cuda.max_memory_allocated()` vs. your estimate (weights + gradients + Adam states + activations)
- [ ] Profile one training step (`torch.profiler`, CPU activities)
- [ ] Thread scaling: tokens/s at 1, 2, 4, 8 and 10 threads. Explain why it doesn't scale linearly.
- [ ] Re-run one config with the same seed and report the variance
- [ ] `experiments/report.md` with limitations

**Ship:** experiment report.

## Week 8: Stage 6a, adapt and evaluate (Kaggle)
- [ ] Choose a **narrow** use case (e.g. a fixed output format, classification, short structured extraction); write the "prompting vs RAG vs fine-tuning" justification
- [ ] Golden eval set (50–100 items) frozen *before* tuning, with failure categories; prefer automatic metrics (exact match, format validity, F1)
- [ ] Laptop: write the training and eval scripts and dry-run them on SmolLM2-135M for 20 steps
- [ ] **Kaggle:** baseline Qwen2.5-0.5B-Instruct (or 1.5B) with prompting only, evaluated on the golden set
- [ ] **Kaggle:** LoRA SFT (fp16, rank 8–16, seq 512, 2–5k examples), then evaluate before and after
- [ ] **Kaggle:** QLoRA (4-bit base via bitsandbytes) on the same data. Compare memory, speed and quality against LoRA.
- [ ] LLM-as-judge (a larger model on Kaggle, or Ollama on the laptop), checked against your own labels on 20 items
- [ ] Optional, **Kaggle:** small DPO demo with ~500 preference pairs (TRL)
- [ ] Download the LoRA adapter (a few MB) and run it on the laptop to confirm it works on CPU
- [ ] *CPU fallback:* LoRA on SmolLM2-135M in fp32, 500–1,000 examples, seq ≤ 256, ~30–90 min

**Ship:** before/after table including failure cases.

## Week 9: Stage 6b, inference and serving
- [ ] Your KV cache on your GPT: same greedy tokens, speedup measured on CPU
- [ ] Latency, TTFT and throughput vs. batch size and prompt length on CPU (documented conditions)
- [ ] Quantization on CPU: (a) `torch.ao.quantization.quantize_dynamic` int8 on your GPT/SmolLM; (b) the same model in Ollama as Q8 vs Q4 GGUF. Compare memory, tokens/s and eval quality.
- [ ] FastAPI service: validation, errors, logging, model version, tests, Dockerfile (a CPU image)
- [ ] **Kaggle:** vLLM serving the Week 8 model. Measure throughput at 1, 8 and 32 concurrent requests vs. plain HF `generate`, and explain continuous batching and PagedAttention against the numbers.
- [ ] Notes: speculative decoding

**Ship:** benchmark table plus a running API.

## Week 10: RAG prototype
- [ ] Ingest → chunk → embed (small CPU embedder: `nomic-embed-text` via Ollama or `bge-small`) → vector store (FAISS-cpu/Chroma)
- [ ] Hybrid retrieval (BM25 + vectors) and reranking (small cross-encoder, top-20 → top-5 keeps CPU latency acceptable)
- [ ] Cited answers via Ollama (qwen2.5:1.5b/3b Q4); keep the corpus to ≤ ~200 docs
- [ ] Eval: recall@k, MRR, faithfulness, and "I don't know" behavior
- [ ] Per-user document access filter, plus prompt-injection test documents

**Ship:** RAG eval table plus a demo.

## Week 11: Architect capstone
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
