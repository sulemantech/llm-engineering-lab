# Attention Is All You Need: Notes

## Problem the paper solves (why not RNNs?)

<!-- TODO: write 4–5 summary bullets here in your own words, using the self-check below. -->

### Self-check: RNN and LSTM (2026-11-03), first attempt 7/12

**Q1. How an RNN works.** An RNN reads "the cat sat on the mat" one word at a time. What does it carry from one step to the next, and what does it do with each new word?
- *My answer:* RNN read a word sequentially one by one and update its memory. It carries memory reference and update it with every new word it reads.
- *Score:* 2/2
- *Refined:* The memory is the **hidden state**, a fixed-size vector updated at every step with the same weights: `h_t = f(h_{t-1}, x_t)`.

**Q2. The speed problem.** Why can't an RNN process all six words at the same time, even on a GPU with thousands of cores?
- *My answer:* Because its mechanism is sequential it is not made for parallel processing; it reads a word and updates its memory.
- *Score:* 1/2
- *Refined:* It's a **dependency**: step t needs `h_{t-1}`, which needs `h_{t-2}`, and so on. Step 5 can't start until step 4 finishes, so extra GPU cores sit idle. Training time grows with sequence length no matter the hardware.

**Q3. The memory problem.** In "The **keys** that I left on the kitchen table yesterday **are** missing", why is it hard for an RNN to connect "keys" to "are" 10 steps apart?
- *My answer:* After every word it updates the memory, and every update dilutes the previous references to words and their connections. After many words, the connection becomes loosely coupled.
- *Score:* 2/2
- *Refined:* The whole past is squeezed into one fixed-size vector, and every new word overwrites part of it: the **information bottleneck**.

**Q4. Vanishing gradients.** During training the error signal flows backwards through every time step. What happens to it over 50–100 steps, and what does that mean for long-range learning?
- *My answer:* Didn't know.
- *Score:* 0/2
- *Refined:* Going backwards, the signal is **multiplied** by a factor at every step. With a factor of 0.9: 0.9¹⁰ ≈ 0.35, 0.9⁵⁰ ≈ 0.005, 0.9¹⁰⁰ ≈ 0.00003. The learning signal from "are" fades out before it reaches "keys", so the model can't learn long-range connections. With a factor of 1.1, it **explodes** instead (1.1⁵⁰ ≈ 117), making training unstable, which is why gradient clipping exists.

**Q5. What LSTMs fixed, and what they didn't.** An LSTM adds a cell state and forget/input/output gates. Which problems does that help with, and which does it not solve?
- *My answer:* LSTM fixed the referencing and connection of words through long short-term memory, but it still couldn't solve the sequential limitation of the RNN.
- *Score:* 1/2
- *Refined:* **How it helps:** the cell state carries information forward by *adding* updates (controlled by the gates) instead of repeatedly multiplying, so information and gradients survive much longer (Q3 and Q4 improved). **Still unsolved:** it's still sequential (Q2), still a fixed-size bottleneck, and still struggles over hundreds of tokens.

**Q6. Connecting to the paper.** Attention lets every word look directly at every other word in one step. Which problems does that fix, and what new cost might it introduce?
- *My answer:* It fixed all the problems above in the RNN and LSTM. The cost is the parallelism it introduced, but we were ready with the introduction of GPUs.
- *Score:* 1/2
- *Refined:* It fixes Q2 (all positions are computed in parallel), Q3 (no bottleneck: every word can be looked at directly) and Q4 (the path between any two words is 1 step). Parallelism is the **benefit**, not the cost. The real costs:
  1. **Quadratic cost:** n words means n² comparisons (1,000 words → 1 million; 10,000 → 100 million). Doubling the context quadruples the attention compute and memory.
  2. **No sense of order:** attention sees all words at once, so "dog bites man" and "man bites dog" look the same without help. The fix is **positional encoding**.

### Retake (to do)
- [ ] **Q4:** in 1–2 sentences, why can't an RNN learn that "keys" relates to "are" 10 words later?
- [ ] **Q6:** name the two new problems attention introduces.

## Architecture overview (diagram)

## Scaled dot-product attention: shapes at each step

## Why scale by sqrt(d_k)?

## Multi-head attention: what it adds

## Positional encoding

## Training setup (data, optimizer, LR schedule, regularization)

## Results and limitations

## Open questions
