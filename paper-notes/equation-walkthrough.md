# Equation → Code → Test

| Paper equation / section | Code (file:function) | Test that verifies it |
|---|---|---|
| Attention(Q,K,V) = softmax(QKᵀ/√d_k)V (§3.2.1) | `src/llmlab/attention/core.py:scaled_dot_product_attention` | `tests/test_stage2_attention.py` |
| MultiHead (§3.2.2) | `src/llmlab/attention/mha.py` | |
| PE (§3.5) | `src/llmlab/attention/positional.py` | |
