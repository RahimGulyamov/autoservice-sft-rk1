---
base_model: Qwen/Qwen2.5-3B-Instruct
library_name: peft
language:
- ru
license: other
license_name: qwen-research
license_link: LICENSE-QWEN.txt
---

# Autoservice RK1 LoRA · Built with Qwen

Educational non-commercial experiment by Гулямов Рахим, prepared with AI assistance,
trained on 2026-11-05 on Tesla T4.

- Base: Qwen/Qwen2.5-3B-Instruct, revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`.
- Adapter: rank 8, alpha 16, dropout 0.05, q_proj/v_proj, 1,843,200 trainable parameters.
- Training: 100 synthetic dialogues / 20 scenarios, 50 optimizer steps, effective batch 4,
  LR 2e-4, FP16 compute over 4-bit NF4 base, seed 42.
- Dev answer-only loss: 2.1696 → 1.3106.
- Held-out strict structure adherence: 100% → 62.5%; complete format contract: 83.3% → 50.0%.
- Author-defined urgency accuracy: 66.7% → 79.2%; the confidence interval includes zero change.

The adapter is published for reproduction of a first experiment, with a negative result
on the primary metric. It is not an improved vehicle diagnostic system. Both the base
and adapted model made content errors; the model judge missed some important ones.

Data: `../../data/`. Report: `../../REPORT.md`. Environment and revisions: `../`.
Load with the exact base revision and NF4 settings from `../../experiment.py`;
`python experiment.py after` in the repository root evaluates the saved adapter.
The adapter alone is not a standalone full model.

See LICENSE-QWEN.txt, Notice and MODIFICATIONS.md. Tokenizer files come from the base
model; the adapter matrices are newly trained modifications.
