# Running on a free GPU (Kaggle / Colab)

Develop and test locally; use the GPU only for heavy runs (Stage 5 training, Stage 7 LoRA/DPO).

1. Push your latest code to GitHub.
2. Kaggle → New Notebook → Settings → Accelerator: **GPU T4** (internet on).
3. In the first cell:
   ```bash
   !git clone https://github.com/<you>/llm-engineering-lab.git
   %cd llm-engineering-lab
   !pip install -e . --no-deps pyyaml tqdm   # keep Kaggle's CUDA torch
   !python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```
4. Run your training script with a config from `experiments/configs/`.
5. Download `experiments/results/runs.jsonl` (and checkpoints if needed), then commit the results locally.

Kaggle gives ~30 GPU h/week; sessions time out, so always checkpoint and support resume.
