# NULLXES / SHINRA 40B MoE — проектная документация

Дата: 24.10.2026. Автор: ASTRA. Authority: NULLXES / Founder.

Комплект D01–D10 подготовлен для founder review согласно V3.0 §19. Математический аудит D02 принят; новые системные решения и неутверждённые предложения сохраняют свои статусы. Создание документов не является разрешением на исполнение модели.

| Документ | Статус |
|---|---|
| [D01 — Architecture](D01_ARCHITECTURE.md) | Baseline и системные контракты; initialization UNKNOWN |
| [D02 — Parameters](D02_PARAMETERS.md) | Математический аудит принят Engineering Order 003 |
| [D03 — Routing](D03_ROUTING.md) | Routing invariants V3.0; coefficients PROPOSED |
| [D04 — Attention/context](D04_ATTENTION_CONTEXT.md) | Размерности утверждены; RoPE policy/curriculum PROPOSED |
| [D05 — Tokenizer](D05_TOKENIZER.md) | Vocabulary size утверждён; IDs/алгоритмические детали PROPOSED |
| [D06 — Data pipeline](D06_DATA_PIPELINE.md) | Контракты PROPOSED; corpus NOT APPROVED |
| [D07 — Pretraining](D07_PRETRAINING.md) | Training/distributed/checkpoint contracts PROPOSED |
| [D08 — Project structure](D08_PROJECT_STRUCTURE.md) | Целевое дерево, не существующие исходники |
| [D09 — Dependencies](D09_DEPENDENCIES.md) | Источники проверены; actual GM compatibility UNKNOWN |
| [D10 — Decisions](D10_DECISIONS.md) | Реестр approved/proposed/unknown и review items |

Дополнительно: [Module contracts](MODULE_CONTRACTS.md), [Implementation status](IMPLEMENTATION_STATUS.md), [Architecture changelog](ARCHITECTURE_CHANGELOG.md).

## Статусы

- VERIFIED — подтверждено распоряжением основателя, доступным материалом или первичным источником. Не означает проверку runtime.
- CALCULATED — аналитический расчёт с явными предпосылками.
- PROPOSED — предложение, не выданное за утверждённое решение.
- UNKNOWN — сведения отсутствуют.
- BLOCKED — операция требует решения, ресурса или отдельного разрешения.

## Обязательные границы

SHINRA — самостоятельная decoder-only causal LLM с MoE в каждом из 48 блоков. Чужие pretrained weights, checkpoints и архитектура старой SHINRA не используются. Dense FFN fallback и shared experts запрещены. Размерности не меняются самостоятельно.

Текущая стадия: DOCUMENTATION REVIEW по V3.0. Сначала завершить и сверить D01–D10, затем получить отдельное подтверждение реализации (§19). Test cases, синтетические данные, локальная и удалённая инициализация/обучение/inference, установки и автоматические скачивания не разрешены.

Корпус: NOT SELECTED. Готовые массовые pretraining corpora не подключаются. GPU planning, VRAM-бюджеты, назначение устройств и топологии находятся вне области работы ASTRA. Известны только две машины для training и одна для initialization; их конфигурации UNKNOWN.

## Состояние артефактов

COMPLETED: подготовлен комплект D01–D10 с MODULE_CONTRACTS, IMPLEMENTATION_STATUS и ARCHITECTURE_CHANGELOG. FILES CHANGED: только Markdown в docs. NOT EXECUTED: модель, tokenizer training, тесты, установки, скачивания и удалённые вычисления. Нет заявлений о работоспособности, производительности, качестве модели или освоенном 256K-контексте.

Ожидается founder review комплекта и отдельное подтверждение перехода к исходникам. Сохранение документации не переводит PROPOSED в утверждённое состояние.
