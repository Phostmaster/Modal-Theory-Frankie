Frankie v7
Neuro-Symbolic Conversational Architecture with Dynamic Compute Allocation, Selective Memory, and Measured Efficiency Gains

Overview:
Frankie is a neuro-symbolic conversational system that treats compute, memory, and internal processing as privileged resources. Instead of the standard “dump everything into context every turn” approach, Frankie uses a lightweight controller to route turns, selectively retrieve memory, and activate expensive operations only when they are genuinely needed.

The architecture was designed and benchmarked to answer one practical question:
Can we maintain (or improve) conversational quality while significantly reducing token usage, latency, and overall compute cost?

Key Features
Dynamic mode routing (Home / Analytic / Engagement) with hysteresis for stability
Privilege Vector for fine-grained control of tokens, temperature, memory depth, and internal operations
Selective memory retrieval (only when needed)
Candidate reply system (gated, used only on high-warmth / unstable turns)
Lightweight LoRA specialization for analytic and engagement modes
Internal MT-inspired modules (phase-weighted attention, hidden-state evolution) in v7
Verifier and tool usage only under strict conditions
Benchmark Results (vs monolithic baseline using same base model)
Across mixed decision/support, emotional-only, and analytic conversations:

Token reduction: ~30–37%
Latency reduction: ~40–47%
Overall work reduction: ~22–27%
Frankie consistently delivers more natural continuity and tone while using noticeably less compute.

Current Status
v7 is the latest frozen version. It includes internal MT/GCF-inspired modifications (phase-weighted attention and hidden-state evolution) while retaining the proven external controller from v5/v6.

Repository Purpose
This repo serves as the public engineering record of Frankie. It contains:

The complete, runnable system
Benchmark scripts and results
Frozen master versions for reproducibility
Future Direction
Further work will explore deeper integration of Modal Theory / Global Coherence Framework (MT/GCF) principles directly into the transformer’s internal dynamics (hidden states, KV-cache persistence, and attention).

How to Run
(See the benchmark_energy_meter.py and main orchestrator files for current test setups.)

License
All rights reserved.
This is an active research / engineering project. Contact for licensing or collaboration enquiries.
