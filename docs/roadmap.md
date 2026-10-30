# From Attention Is All You Need to Production-Ready LLM Engineering

**Audience:** AI Engineer + AI Architect\
**Pace:** 2--3 hours/day; 49-day target, not a deadline.\
**Goal:** Understand the original Transformer paper, implement its core
architecture, build a small GPT-style language model, and demonstrate
practical LLM engineering and architecture judgment.

## How to use this roadmap

Every stage has a **proof-of-understanding gate**. Do not mark a stage
complete just because the code runs. Pass the gate without following a
tutorial or reading your notes. If you fail, record the gap, revisit the
relevant material, and retake the gate with a new example.

Daily rhythm: - 20--30 min: read the paper/course material. - 20--30
min: derive equations, draw diagrams, or trace tensor shapes. - 60--90
min: implement and test. - 15--20 min: explain what you learned and log
results.

Rules: 1. Implement core operations yourself before consulting a
complete reference implementation. 2. Explain important tensor shapes
and masking behavior. 3. Test with tiny controlled examples before
scaling up. 4. Record failed experiments and fixes. 5. Distinguish
measured results, estimates, and proposed designs.

> The original 2017 paper describes an encoder--decoder Transformer for
> sequence-to-sequence tasks. GPT-style LLMs commonly use decoder-only
> Transformers trained for next-token prediction. This course implements
> the original architecture and then builds GPT as the main
> language-model project. A small GPT proves foundational understanding;
> it is not a competitive general-purpose LLM.

## Outcomes

By the end, you should be able to: - Explain the original paper's
motivation, architecture, equations, training setup, and trade-offs. -
Implement attention, multi-head attention, positional encoding,
feed-forward networks, residual paths, normalization, encoder, and
decoder. - Build, train, save, reload, and generate text with a small
GPT-style model. - Diagnose basic training problems and produce
reproducible experiments. - Explain tokenization, pretraining,
instruction fine-tuning, evaluation, KV cache, quantization, serving,
and cost/latency trade-offs. - Defend an LLM architecture using explicit
assumptions, evidence, alternatives, and risks.

## Suggested repository structure

``` text
transformer-llm-engineering/
├── README.md
├── paper-notes/
│   ├── attention-is-all-you-need.md
│   └── equation-walkthrough.md
├── src/
│   ├── tokenizer/
│   ├── attention/
│   ├── transformer/
│   ├── gpt/
│   ├── training/
│   ├── generation/
│   └── serving/
├── tests/
├── experiments/
│   ├── configs/
│   ├── results/
│   └── report.md
├── evaluation/
├── deployment/
└── architecture/
    ├── system-diagram.md
    ├── capacity-estimates.md
    ├── risk-register.md
    └── adr/
```

# Stage 0 --- Environment and baseline (Day 1)

**Tasks** - Set up Python, PyTorch, Git, a virtual environment, and
tests. - Review tensors, matrix multiplication, broadcasting, indexing,
and autograd. - Compute gradients for a simple scalar function. - Record
library versions and CPU/GPU availability.

**Deliverables:** Reproducible setup instructions, tensor examples,
gradient example.

**Proof-of-understanding gate** - Explain matrix-multiplication output
shapes before running the code. - Explain tensor shape vs. tensor
values, gradient vs. parameter, and optimizer vs. loss. - Recreate the
environment and rerun the examples.

**Pass:** Setup is reproducible; examples run; shape predictions and
explanations are correct.

# Stage 1 --- Language modelling fundamentals (Days 2--7)

  -----------------------------------------------------------------------
  Day                                 Study and build
  ----------------------------------- -----------------------------------
  2                                   Text, vocabulary, token IDs, input
                                      sequences, next-token targets;
                                      implement character tokenizer and
                                      encode/decode tests.

  3                                   Frequency baseline and bigram
                                      model; sample next
                                      characters/tokens.

  4                                   Logits, softmax, negative
                                      log-likelihood, cross-entropy;
                                      calculate a tiny example by hand
                                      and verify with PyTorch.

  5                                   Embeddings and a small neural
                                      language model; produce next-token
                                      logits.

  6                                   Backpropagation, gradient descent,
                                      optimizers, learning rate; train on
                                      a controlled corpus.

  7                                   Train/validation split, sampling,
                                      random seed, basic experiment log.
  -----------------------------------------------------------------------

**Deliverables:** Character tokenizer, bigram baseline, neural baseline,
training loop, sample generations, explanation of the objective.

**Proof-of-understanding gate** 1. Given a short sequence, construct
inputs and shifted targets. 2. Explain token IDs vs. logits
vs. probabilities vs. labels. 3. Explain why cross-entropy rewards
probability assigned to the correct target. 4. Build batches and targets
for a new tiny corpus without copying a tutorial. 5. Test
encoding/decoding, target alignment, output dimensions, and finite loss.
6. Intentionally break target alignment or learning rate, diagnose the
symptom, and fix it. 7. Explain why training loss alone does not prove
generalization.

**Pass:** Tests pass, loss decreases on a controlled task, and you can
explain the data-to-loss path end to end.

# Stage 2 --- Read and implement attention (Days 8--14)

**Primary source:** [Attention Is All You
Need](https://arxiv.org/abs/1706.03762), especially the architecture and
attention sections.

  -----------------------------------------------------------------------
  Day                                 Study and build
  ----------------------------------- -----------------------------------
  8                                   Read abstract, introduction, and
                                      architecture overview; summarize
                                      why the paper proposes attention
                                      instead of recurrence.

  9                                   Implement Q/K/V projections,
                                      attention scores, scaling, softmax,
                                      and weighted sum.

  10                                  Annotate tensor shapes; test
                                      multiple batch sizes and sequence
                                      lengths.

  11                                  Implement causal and padding masks;
                                      test masked positions.

  12                                  Implement multi-head splitting,
                                      parallel attention, concatenation,
                                      and output projection.

  13                                  Implement sinusoidal positional
                                      encoding and inspect values by
                                      position.

  14                                  Write equation-by-equation notes
                                      linking paper to code and tests.
  -----------------------------------------------------------------------

Core equation:

\[ `\operatorname{Attention}`{=tex}(Q,K,V)=
`\operatorname{softmax}`{=tex}`\left`{=tex}(`\frac{QK^\top}{\sqrt{d_k}}`{=tex}`\right`{=tex})V
\]

**Proof-of-understanding gate** - Derive score/output shapes from Q, K,
V shapes. - Explain query, key, value, scaling by (`\sqrt{d_k}`{=tex}),
softmax, and weighted aggregation. - Implement attention from a blank
file before looking at your existing version. - Compare a tiny
calculation with a manual result. - Verify masking prevents future-token
information from affecting causal predictions. - Test at least two
sequence lengths and confirm gradients are finite and nonzero where
expected.

**Pass:** Manual calculation agrees within numerical tolerance, shape
and masking tests pass, and you can explain the operation in your own
words.

# Stage 3 --- Reconstruct the original Transformer (Days 15--21)

  -----------------------------------------------------------------------
  Day                                 Study and build
  ----------------------------------- -----------------------------------
  15                                  Implement the position-wise
                                      feed-forward network.

  16                                  Implement residual connections and
                                      layer normalization; document the
                                      original paper's sublayer ordering.

  17                                  Build one encoder layer from
                                      self-attention and feed-forward
                                      sublayers.

  18                                  Stack encoder layers; add
                                      embeddings and positional encoding.

  19                                  Build decoder masked self-attention
                                      and encoder--decoder
                                      cross-attention.

  20                                  Complete decoder stack, output
                                      projection, target shifting, and
                                      sequence loss.

  21                                  Train on a tiny synthetic
                                      sequence-mapping task; test masks
                                      and forward-pass shapes.
  -----------------------------------------------------------------------

**Deliverables:** Encoder, decoder, full forward pass,
shape/mask/gradient tests, architecture diagram, paper notes.

**Proof-of-understanding gate** - Draw input-to-output data flow and
explain encoder self-attention, decoder masked self-attention, and
cross-attention. - Explain residual connections, layer normalization,
feed-forward layers, source padding masks, and target causal masks. -
Change layer count or head count without breaking dimensions. - Trace
one batch and report shapes at each major step. - Verify future target
tokens are masked and the decoder can use encoder outputs. - Overfit a
tiny synthetic task and explain why this does not prove large-scale
translation quality.

**Pass:** Diagram matches code, tests pass, and the model learns a
controlled task. You can clearly distinguish the original
encoder--decoder model from GPT.

# Stage 4 --- Build GPT from scratch (Days 22--28)

  -----------------------------------------------------------------------
  Day                                 Study and build
  ----------------------------------- -----------------------------------
  22                                  Define vocabulary, context length,
                                      embedding dimension, layer count,
                                      and head count; draw the model.

  23                                  Implement token embeddings,
                                      positional information, and causal
                                      self-attention.

  24                                  Add feed-forward block,
                                      normalization, and residual paths.

  25                                  Stack blocks; implement logits and
                                      next-token loss.

  26                                  Add training, validation, and
                                      checkpoint save/load.

  27                                  Implement autoregressive
                                      generation, greedy decoding,
                                      temperature, and top-k sampling.

  28                                  Run end-to-end training, reload a
                                      checkpoint, and generate from a
                                      prompt.
  -----------------------------------------------------------------------

**References:** [Karpathy --- Let's build
GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY),
[nanoGPT](https://github.com/karpathy/nanoGPT). Use references to
verify, not to skip understanding.

**Proof-of-understanding gate** - Explain original encoder--decoder
Transformer vs. decoder-only GPT. - Explain causal masking and target
shifting. - Rebuild a minimal GPT block from your own modules. - Change
context length or head count and explain required shape changes. -
Verify no future-token leakage, finite loss/gradients, checkpoint
reload, and deterministic output under controlled settings. - Diagnose a
deliberately introduced target-shift or mask-orientation bug.

**Pass:** The training and generation pipeline works; tests demonstrate
causal behavior; you can explain each module's purpose. Polished prose
is not required.

# Stage 5 --- Training, data, experiments, and debugging (Days 29--35)

  -----------------------------------------------------------------------
  Day                                 Study and build
  ----------------------------------- -----------------------------------
  29                                  Data cleaning, deduplication,
                                      train/validation split, sequence
                                      packing; document data provenance.

  30                                  Plot train/validation loss; explain
                                      the gap and overfitting symptoms.

  31                                  Change one major hyperparameter at
                                      a time; log results.

  32                                  Compare character and subword
                                      tokenization, vocabulary size, and
                                      sequence length.

  33                                  Add gradient clipping,
                                      learning-rate schedule, and
                                      checkpoint resume.

  34                                  Estimate parameters, memory,
                                      compute; profile bottlenecks.

  35                                  Write reproducible experiment
                                      report with limitations.
  -----------------------------------------------------------------------

**Proof-of-understanding gate** - Interpret a loss curve and distinguish
plausible overfitting, underfitting, optimization failure, and data
leakage. - Compare at least two configurations with controlled
conditions. - Record seed, data, tokenizer, model dimensions, context
length, optimizer, learning rate, steps, hardware, and metrics. - Re-run
one configuration and assess reproducibility. - Calculate parameter
count and estimate memory, stating what the estimate omits. - Explain
why inference memory and training memory differ.

**Pass:** Report contains measured results and justified conclusions,
not unsupported claims. Assumptions and limits are explicit.

# Stage 6 --- Modern LLM engineering (Days 36--42)

  -----------------------------------------------------------------------
  Day                                 Study and build
  ----------------------------------- -----------------------------------
  36                                  Subword tokenizers, special tokens,
                                      chat templates, pretraining
                                      vs. instruction tuning.

  37                                  Fine-tune a small pretrained model
                                      with supervised examples; compare
                                      before/after outputs.

  38                                  Create evaluation set, metrics,
                                      human review, failure categories,
                                      regression tests.

  39                                  Benchmark inference latency,
                                      throughput, batching, and
                                      time-to-first-token.

  40                                  Inspect or implement KV caching;
                                      compare cached and uncached
                                      generation.

  41                                  Study quantization and hardware
                                      constraints; compare memory, speed,
                                      and quality where feasible.

  42                                  Wrap model in an API with input
                                      validation, logs, errors,
                                      versioning, and tests.
  -----------------------------------------------------------------------

**Reference:** [Hugging Face LLM
Course](https://huggingface.co/learn/llm-course/).

**Proof-of-understanding gate** - For a realistic use case, justify
prompting, RAG, fine-tuning, continued pretraining, or training from
scratch. - Create an evaluation set and compare a baseline with an
adapted model, including failure cases. - Explain metric limitations and
avoid cherry-picked examples. - Measure latency/throughput under
documented conditions. - Explain KV caching, batching, and quantization
trade-offs. - Demonstrate API validation and error handling. - If
hardware prevents an experiment, document the method and run a smaller
feasible demonstration; do not claim unrun results.

**Pass:** You can justify the adaptation approach, reproduce evaluation,
and explain benchmark limitations.

# Stage 7 --- AI Architect capstone and hiring evidence (Days 43--49)

Use this case: *Design an internal enterprise knowledge assistant with
sensitive documents, measurable quality requirements, and a fixed
latency/cost budget.*

  -----------------------------------------------------------------------
  Day                     Focus                   Deliverable
  ----------------------- ----------------------- -----------------------
  43                      Hosted API vs. open     Decision matrix with
                          model; build vs. buy;   assumptions
                          RAG vs. fine-tuning     
                          vs. pretraining         

  44                      Quality, robustness,    Evaluation and
                          safety, leakage,        acceptance plan
                          release evaluation      

  45                      Serving, batching,      Deployment diagram and
                          queues, autoscaling,    request flow
                          fallback, capacity      

  46                      Observability,          Metrics and alerting
                          cost/request,           plan
                          model/version tracking, 
                          alerts                  

  47                      Privacy, access         Risk register and
                          control, licensing,     mitigations
                          prompt injection where  
                          relevant, auditability  

  48                      Cost/capacity           Architecture decision
                          estimates, failure      record
                          modes, rollout and      
                          rollback                

  49                      Polish repository,      Portfolio-ready project
                          demo, report, and       
                          walkthrough             
  -----------------------------------------------------------------------

**Proof-of-understanding gate** 1. State functional and non-functional
requirements and measurable success criteria. 2. Justify model strategy
and explain what evidence could change your decision. 3. Draw request
flow, model serving, retrieval if applicable, identity/access controls,
data boundaries, monitoring, and failure handling. 4. Define offline
evaluation, regression tests, safety checks, and release gates. 5.
Estimate request volume, tokens, throughput, latency, and cost drivers;
label assumptions. 6. Document privacy, licensing, prompt injection
where relevant, hallucination, outage, and vendor risks. 7. Compare at
least two viable designs and defend the selected option. 8. Link tested
code, evaluation, and benchmark evidence; clearly separate
implementation from proposed production design.

**Pass:** Another engineer or architect can challenge the assumptions
and you can defend decisions using evidence, trade-offs, and explicit
limitations.

------------------------------------------------------------------------

## Stage-gate scoring

Score each category from 0 to 2:

-   **0 --- Not demonstrated:** explanation or evidence missing.
-   **1 --- Partial:** needs hints or cannot verify independently.
-   **2 --- Demonstrated:** explains in own words, implements/reasons
    independently, and verifies with evidence.

Score four categories per stage: 1. Conceptual explanation. 2.
Independent implementation or design. 3. Tests and evidence. 4.
Debugging and trade-off reasoning.

**Pass threshold:** At least 7/8 and no zero in any category. For
attention math, causal masking, target alignment, and evaluation
integrity, require 2/2 before advancing. Retake failed categories after
targeted practice.

## Interview readiness checklist

### Transformer fundamentals

-   Why did the paper propose a Transformer?
-   Derive scaled dot-product attention and explain tensor shapes.
-   What does multi-head attention add?
-   How does encoder--decoder differ from GPT?
-   Why are causal masks and target shifting necessary?
-   What do residual connections, normalization, and feed-forward blocks
    contribute?

### Training and model engineering

-   How do logits become loss, and how does backpropagation update
    parameters?
-   How do you distinguish overfitting from optimization failure?
-   What changes with model width, depth, context length, or batch size?
-   How do tokenization and data quality affect training?
-   What does a small from-scratch model prove---and not prove?

### AI architecture

-   When would you choose RAG, fine-tuning, hosted APIs, or self-hosted
    inference?
-   How would you evaluate quality and detect regressions?
-   How would you estimate latency, throughput, GPU memory, and cost?
-   How do KV cache, batching, and quantization affect inference?
-   How would you design access controls, observability, rollout, and
    rollback?

Use your repository, measurements, diagrams, and trade-off records as
evidence instead of memorized answers.

## Primary resources

1.  [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2.  [Andrej Karpathy --- Let's build
    GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY)
3.  [nanoGPT](https://github.com/karpathy/nanoGPT)
4.  [The Annotated
    Transformer](https://nlp.seas.harvard.edu/annotated-transformer/)
5.  [Hugging Face LLM Course](https://huggingface.co/learn/llm-course/)

## Final definition of done

You are finished when you can demonstrate that you can: - Explain the
paper's design and equations in your own words. - Implement and test its
main components and a small encoder--decoder Transformer. - Build,
train, save, reload, and generate from a GPT-style model. - Diagnose
masking and training bugs. - Produce reproducible experiments and honest
performance reports. - Adapt a pretrained model and evaluate it. -
Benchmark and expose inference through a service. - Defend an enterprise
LLM architecture with explicit assumptions, quality criteria, costs,
risks, and alternatives.

The objective is not to rush through 49 days. It is to explain what you
built, prove that it works, understand where it fails, and make sound
engineering and architecture decisions.
