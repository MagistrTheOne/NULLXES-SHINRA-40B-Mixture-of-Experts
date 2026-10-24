# Architecture changelog

## 24.10.2026 — документационная сверка V3.0

VERIFIED: до этой работы существовали только docs/README.md и D02–D05 под описательными именами. D01/D06–D10 не объявлялись ранее существующими.

Переименованы с сохранением тела:

- D02_parameter_accounting.md → D02_PARAMETERS.md.
- D03_moe_routing.md → D03_ROUTING.md.
- D04_attention_long_context.md → D04_ATTENTION_CONTEXT.md.
- D05_native_tokenizer.md → D05_TOKENIZER.md.

Созданы D01_ARCHITECTURE, D06_DATA_PIPELINE, D07_PRETRAINING, D08_PROJECT_STRUCTURE, D09_DEPENDENCIES, D10_DECISIONS, MODULE_CONTRACTS, IMPLEMENTATION_STATUS и этот журнал (все .md). README обновлён под текущий комплект и V3.0 §19.

Утверждённые размеры и D02 total/active не менялись. Новые config/data/training/distributed/checkpoint/inference contracts — PROPOSED. Не утверждены автоматически balancing coefficients, численная RoPE policy, tokenizer IDs, initialization recipe и runtime dependencies. Это дополнение проектных документов, не изменение реализованной модели: исходников нет.

NOT EXECUTED: model initialization/forward/backward/training/inference, dependency installation, downloads, tests, deployment. Hardware planning не выполнялся.
