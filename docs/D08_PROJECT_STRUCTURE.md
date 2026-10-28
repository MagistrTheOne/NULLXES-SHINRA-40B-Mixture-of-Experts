# D08 — Project structure

25.10.2026. VERIFIED: целевая структура V3.0; фактический исходный набор частично реализован. Текущий inventory и blockers приведены в [IMPLEMENTATION_STATUS](IMPLEMENTATION_STATUS.md) и [Engineering Order 005 audit](ENGINEERING_ORDER_005_AUDIT.md).

```text
SHINRA-40B-MoE/
  docs/
    D01_ARCHITECTURE.md
    D02_PARAMETERS.md
    D03_ROUTING.md
    D04_ATTENTION_CONTEXT.md
    D05_TOKENIZER.md
    D06_DATA_PIPELINE.md
    D07_PRETRAINING.md
    D08_PROJECT_STRUCTURE.md
    D09_DEPENDENCIES.md
    D10_DECISIONS.md
    MODULE_CONTRACTS.md
    IMPLEMENTATION_STATUS.md
    ARCHITECTURE_CHANGELOG.md
    README.md
  configs/
    shinra_40b_moe.yaml
  src/
    shinra/
      __init__.py
      core/
        __init__.py
        configuration.py
        validation.py
        initialization.py
      model/
        __init__.py
        embeddings.py
        normalization.py
        rope.py
        attention.py
        decoder.py
        modeling.py
        lm_head.py
      moe/
        __init__.py
        router.py
        experts.py
        dispatch.py
        combine.py
        balancing.py
        layer.py
      tokenizer/
        __init__.py
        byte_bpe.py
        vocabulary.py
        merges.py
        special_tokens.py
        serialization.py
        streaming.py
      data/
        __init__.py
        registry.py
        provenance.py
        ingestion.py
        deduplication.py
        language.py
        packing.py
        streaming.py
      training/
        __init__.py
        objective.py
        optimizer.py
        scheduler.py
        precision.py
        engine.py
        state.py
        telemetry.py
      distributed/
        __init__.py
        ownership.py
        parallelism.py
        expert_parallel.py
        context_parallel.py
        communication.py
      checkpoint/
        __init__.py
        manifest.py
        serialization.py
        transactions.py
        recovery.py
      inference/
        __init__.py
        generation.py
        sampling.py
        kv_cache.py
        decoding.py
  requirements.txt
  pyproject.toml
  README.md
```

## PROPOSED — dependency boundaries

core.configuration — типы и parser; core.validation зависит от этих типов, configuration не импортирует validation обратно. core.initialization получает проверенный план, не импортирует model для обнаружения параметров. Model assembly передаёт tensors/metadata явно.

Tokenizer использует core config, собственные leaf vocabulary/merges/special modules и stdlib; data использует tokenizer interfaces, но tokenizer не импортирует data. Model math и moe primitives используют core, не training/inference. moe.layer получает dispatch/communication adapter через аргументы. distributed содержит logical adapters и не импортирует model assembly/training engine. checkpoint принимает state mappings, не импортирует trainer. training.engine и inference.generation — верхние orchestrators; нижние слои не импортируют их.

Cache protocol передаётся attention как объект явного контракта; attention не импортирует inference.generation. Runtime annotations не создают обратных imports. __init__.py не выполняет model construction, filesystem/network I/O и eager heavy imports. Runtime adapters не выбираются по случайной доступности import.

## Module responsibilities

Подробный mapping всех модулей: [MODULE_CONTRACTS](MODULE_CONTRACTS.md). Будущие pyproject/requirements не дублируют списки зависимостей: один requirements — direct runtime requirements кроме GM; packaging build backend отдельно документируется D09. Не создавать файлов ради дерева, пустых implementation modules или дублирующих версий.

## ACTUAL SOURCE STATUS — 26.10.2026

All listed packages now exist: 59 Python files including package markers. This is file/source presence, not full system readiness. Tokenizer/data implementations, fresh-run DP training, distributed primitives and single-sequence inference have been added. Complete EP/TP/CP model assembly and coordinated training restore remain absent. See [implementation report](ORDER_007_IMPLEMENTATION.md).

Checkpoint transaction producers now use function-local imports of core initialization/training precision; neither imports checkpoint. Model math does not import inference or training. Inference orchestration reuses canonical blocks and attention projections.

## UNKNOWN / BLOCKED

Actual GM compatibility, numerical execution, storage durability, complete distributed assembly and training resume are unverified or incomplete. Historical Order 005 reports describe the earlier 25-file revision.

## Order 008 current override

EP ownership now reaches canonical model/MoE/training modules. CP reaches model attention. Complete-set DP/EP restore and publication source exists in existing checkpoint modules. Earlier absent-integration statements describe Order 007; TP and CP training remain outstanding. No duplicate model files or additional packages created.

## Order 009 continuation — 27.10.2026

Canonical TP model source and full-logical-stream slice initialization are now implemented, with explicitly replicated router/norms. CP real-target boundary partitioning and summed objective added. TP training/checkpoint integration, full CP training, uneven EP scheduling and mixed-group orchestration remain incomplete and fail closed where exposed. Source Complete is NOT achieved. No model/test/checkpoint execution. Details and changed files: [Order 009 report](ORDER_009_DISTRIBUTED_COMPLETION.md).


## Current mixed integration — 27.10.2026

Supersedes earlier incomplete-integration status entries above: canonical TP, CP training with real boundary targets, uneven EP zero-work participation, explicit mixed-group gradient/clip ownership and mixed Safetensors snapshot/publication/resume are implemented in existing modules. ParallelLayout and MixedExecution require approved logical ownership and caller-created real groups; no topology or operational defaults are supplied. Source replicas agree on availability, content and pending cursors; TP copies do not inflate counts. Post-mutation failure requires complete recovery and group restart.

Distributed runtime correctness and actual GM compatibility are UNVERIFIED; execution remains blocked on approved operational settings, corpus/tokenizer, initialization and storage contracts. Resident single-sequence inference remains separate and does not silently load/repartition mixed shards. Detailed source contracts, changed files and limitations: [Order 009 update](ORDER_009_DISTRIBUTED_COMPLETION.md#mixed-integration-update--27102026). Historical audit reports remain unchanged. No model or tests executed.
