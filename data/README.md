# Data (git-ignored)

Record provenance for every dataset you use: source, licence, date downloaded, preprocessing.

| Dataset | Stage | Source | Size |
|---|---|---|---|
| TinyShakespeare | 1–4 | https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt | ~1 MB |
| TinyStories | 5 | https://huggingface.co/datasets/roneneldan/TinyStories | ~2 GB (use a subset on CPU) |

```bash
curl -L -o data/tinyshakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```
