# Multimodal Single-Investor Digital Twin

Reviewer reference code for a founder interacting with one focal investor. The input combines the founder's words, acoustic features, and visible evidence available before each investor response. The output is the investor's next public text response and the resulting on-air negotiation state.

| File | Role |
| --- | --- |
| `reviewer_data_pipeline.py` | Reviewed observations, evidence cutoff, and matched T / T+A / T+V / T+A+V rows. |
| `reviewer_annotation.py` | Source-linked text, audio, video, and outcome codes with blinded adjudication. |
| `reviewer_audio.py` | Founder-only acoustic measurements with source and window checks. |
| `reviewer_memory.py` | Train-only case retrieval and versioned knowledge snapshots. |
| `reviewer_model.py` | LoRA policy, masked completion loss, and multimodal gated prefix. |
| `reviewer_training.py` | Matched supervision, model bindings, and held-out prediction export. |
| `reviewer_dyad.py` | Founder–investor turn state, fact availability, and public offer ledger. |
| `reviewer_orchestrator.py` | Snapshot-bound interaction, evidence traces, and response validation. |
| `reviewer_evaluation.py` | Investor-turn, coded-response, and full-rollout scoring. |
