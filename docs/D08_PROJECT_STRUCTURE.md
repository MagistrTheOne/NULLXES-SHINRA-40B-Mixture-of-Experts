# D08 — Project structure

24.10.2026. VERIFIED: целевая структура V3.0. Фактически сейчас созданы только docs. Ни один показанный Python/YAML файл не объявляется существующим.

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

## UNKNOWN / BLOCKED

Фактический import graph отсутствует до исходников. Проверено направление проектных dependencies, но не отсутствие циклов в ещё не написанном коде. Backend build/packaging choices не утверждены. Реализация ожидает отдельного подтверждения V3.0 §19.

