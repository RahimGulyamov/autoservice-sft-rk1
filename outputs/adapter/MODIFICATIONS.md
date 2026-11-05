# Modified model files · Built with Qwen

`adapter_model.safetensors` contains newly trained LoRA matrices (q_proj/v_proj,
rank 8, alpha 16), not original base weights. Trained on 2026-11-05 for an
educational non-commercial experiment by Гулямов Рахим with AI assistance.

Base: Qwen/Qwen2.5-3B-Instruct. Exact SHA is in ../model_revisions.json
(repository path: outputs/model_revisions.json).
Tokenizer files were copied from the base model for reproducible loading.

This first adapter degraded strict answer-format adherence. It is provided
for reproduction and evaluation, with the Qwen Research License and Notice.
It is not validated for real vehicle diagnosis or deployment.
