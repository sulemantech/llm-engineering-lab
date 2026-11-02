# Running heavy jobs on Kaggle (free T4 GPU)

Build and debug on the laptop. Kaggle runs only scripts that already work on CPU with a tiny config.

## One-time setup
1. Create a Kaggle account → **Settings → Phone verification** (required for GPU and internet).
2. **Settings** shows your weekly GPU quota (~30 h). Check it before planning a big run.
3. If the GitHub repo is private: create a GitHub fine-grained token (read-only, this repo only) and add it to the
   notebook under **Add-ons → Secrets** as `GH_TOKEN`. Never paste tokens into notebook cells.

## Each run
1. Laptop: `uv run pytest`, then a 50-step tiny run on CPU succeeds → `git push`.
2. Kaggle → **New Notebook** → right panel: **Accelerator: GPU T4 x1**, **Internet: on**.
3. First cell:
   ```bash
   # Public repo:
   !git clone https://github.com/<you>/llm-engineering-lab.git
   # Private repo (token from Kaggle Secrets):
   # from kaggle_secrets import UserSecretsClient
   # tok = UserSecretsClient().get_secret("GH_TOKEN")
   # !git clone https://{tok}@github.com/<you>/llm-engineering-lab.git

   %cd llm-engineering-lab
   !pip install -q -e . --no-deps && pip install -q psutil   # keep Kaggle's CUDA torch
   !python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
4. Run the script with a config, writing checkpoints and results into `/kaggle/working`:
   ```bash
   !python -m llmlab.training.train --config experiments/configs/s6-tinystories.yaml --out /kaggle/working/run1
   ```
5. For runs over ~30 min: **Save Version → Save & Run All (Commit)**. It keeps running with the browser closed
   (up to ~12 h). When it finishes, open the version → **Output** → download the files.
6. Laptop: append the downloaded result row to `experiments/results/runs.jsonl`, commit, and log the GPU hours used in `LOG.md`.

## T4 specifics
- Use **fp16**, not bf16 (the T4 has no fast bf16):
  ```python
  scaler = torch.amp.GradScaler("cuda")
  with torch.autocast("cuda", dtype=torch.float16):
      logits, loss = model(x, y)
  scaler.scale(loss).backward()
  scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
  scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
  ```
- 16 GB VRAM: check `torch.cuda.max_memory_allocated()` after a few steps before committing a long run.
- `F.scaled_dot_product_attention` picks fused kernels automatically on GPU.

## Resilience
- Checkpoint every ~10 min to `/kaggle/working/<run>/ckpt.pt` (model, optimizer, scaler, step, RNG states).
- Training scripts take `--resume <path>`. To continue after a timeout, add the previous version's output
  as notebook input (**Add Input → Notebook Output**) and resume from it.
- Large outputs (> a few hundred MB) are slow to download; keep only the best checkpoint plus results.
- Quota or internet problems: run the task's *CPU fallback* from PLAN.md and note it in LOG.md.
