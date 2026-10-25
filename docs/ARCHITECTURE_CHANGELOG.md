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

## 25.10.2026 — Engineering Order 005 static audit

Фактический source inventory обновлён: существуют 20 Python-файлов в `core`, `model` и `moe`; implementation status и D08/D09/module contracts приведены в соответствие с этим источником. Добавлен отдельный отчёт [ENGINEERING_ORDER_005_AUDIT](ENGINEERING_ORDER_005_AUDIT.md).

Исправлены: BF16 weight sampling теперь выполняется в FP32 с явным cast к storage dtype; модельная операция отклоняет UNKNOWN sliding-window policy; `BalancingState` требует явный initial bias вместо скрытого нулевого заполнения. Canonical YAML не менялся: `checkpoint.format` уже равен `safetensors`.

Safetensors serialization, tokenizer, data, training orchestration, distributed, checkpoint transactions/recovery и inference остаются отсутствующими и BLOCKED. Это статический source audit, не runtime validation. Модель, тесты, зависимости, загрузки, deployment и GPU planning не запускались.

## 25.10.2026 — Engineering Order 006, Stage A

Создан `src/shinra/checkpoint/` с manifest, Safetensors serialization, staged transactions и strict recovery. В `requirements.txt` добавлена прямая pin-зависимость `safetensors==0.6.2`; её не устанавливали. Release metadata требует Python >=3.9; Torch extra указывает torch>=1.10, но текущая GM совместимость неизвестна.

Checkpoint API поддерживает exact tensor inventories, model/optimizer/router/RNG группы, именованные optimizer state tensors, JSON metadata, model shard индексы, SHA-256, проверку complete publication и явное требование подтверждённой atomic-replace capability. Durability не обещается. Training resume остаётся BLOCKED до реальных state producers, backend approval и согласования scheduler/cursor/tokenizer/corpus identities. Никаких checkpoint-файлов не создавали.
