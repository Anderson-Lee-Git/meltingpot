# Cluster Guidelines

How to work on the Slurm compute cluster: network access, GPU allocation, and partition choice.

**Tradeoff:** Queue time is the scarce resource here, not GPU time. Spend a minute estimating before allocating rather than sitting in a queue you didn't need to join.

## 1. Compute Nodes Have No Internet

**Anything that needs the network happens on the login node, not the compute node.**

- Compute nodes cannot reach the internet. `pip install`, `huggingface-cli download`, `git fetch`, `wandb` syncs, API calls, etc. will hang or fail if run from a job.
    - Pre-download anything from Hugging Face onto the shared filesystem before submitting jobs.
    - Local download locations:
    - **`<repo-root>/models`** for HF models.
    - **`<repo-root>/datasets`** for HF datasets.
    - Point code at these local paths on the cluster rather than relying on on-the-fly downloads.
- Do network-dependent work on the login node **before** submitting, and point jobs at the results on the shared filesystem
- If a job genuinely must reach the internet at runtime, route the traffic back through the login node over SSH (e.g. an SSH tunnel / reverse proxy from the compute node to the login node). Treat this as the exception, not the default — prefer pre-staging.

The test: A job should run to completion with the network cable unplugged.

## 2. Use `salloc` for Interactive Testing

**When you want a GPU to poke at things, get a persistent interactive allocation.**

- Use `salloc` (not one-off `srun` per command) so the allocation persists across many test commands and you pay queue time once.
- Allocation is not instant — it goes through the queue like any other job. Plan for that; don't request one, discover it's queued, and idle.
- Release it (`exit`) when done so it doesn't count against your limits (see §3).

## 3. Estimate Runtime Before Requesting Time

**The time limit you ask for determines which queue you land in. Pick it deliberately.**

- **Under 1 hour → `gpu-test` bucket.** Much faster queue time. Hard cap: **at most 3 `gpu-test` GPUs per user at a time** (across all your `gpu-test` jobs/allocations combined).
- **1 hour or more → regular queue.** Higher job/GPU limit, but noticeably longer queue time.

Before committing to an allocation:
1. Estimate how long the work actually takes.
2. If it fits in <1h and you have `gpu-test` slots free, request <1h and get the fast queue.
3. If it won't fit, request an honest longer time rather than requesting <1h and getting killed mid-run — a killed run plus a re-queue costs more than one long queue wait.
4. Don't pad a 20-minute job to 4 hours "just in case"; you'll trade a fast queue for a slow one for no reason.

Check what you already hold with `squeue -u $USER` before requesting more, so you don't hit the 3-GPU `gpu-test` cap unexpectedly.

The test: You can say, before typing `salloc`/`sbatch`, roughly how long the job runs and which bucket it will land in.

## 4. Choosing a Partition

**There is no partition that is always faster. Check availability, then decide.**

- **`--partition=ailab` → H200 GPUs.** Shared with everyone else on that partition, so queue time may be higher or lower than default depending on current load — it varies.
- **No `--partition` → other GPU types (e.g. A100), assigned by the scheduler.** Queue time also varies with load.
- Before choosing, check `ailab` availability:
  ```
  shownodes -p ailab
  ```
  If there are idle `ailab` nodes, `--partition=ailab` is likely the fast path *and* gives you H200s. If it's saturated, omit the partition and take whatever the default hands out.
- Only insist on `ailab` when the work actually needs H200-class memory/throughput; otherwise let availability decide.

The test: Your partition choice was made from `shownodes` output, not habit.

---

**These guidelines are working if:** jobs don't fail on network access, you rarely sit in a long queue for a short test, and you can explain why each allocation used the time limit and partition it did.

---

Source: [Anderson-Lee-Git/research-template `.claude/rules/cluster_guideline.md`](https://github.com/Anderson-Lee-Git/research-template/blob/ebd3f936d8a0dcc0ef08366528e601ec3292b939/.claude/rules/cluster_guideline.md) at commit `ebd3f936d8a0dcc0ef08366528e601ec3292b939`.
