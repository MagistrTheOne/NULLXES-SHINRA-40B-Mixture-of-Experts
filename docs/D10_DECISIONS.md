# D10 — Architecture decision register

24.10.2026. Authority: NULLXES. VERIFIED означает подтверждённый источник требования, не runtime correctness.

| ID | Decision | Status / authority | Consequence |
|---|---|---|---|
| A01 | Размерности D02, 39694338560 total, 4304898560 conventional active | VERIFIED: Order 003, V3.0 | Не менять автоматически |
| A02 | MoE во всех 48 blocks, 128 experts, Top-8, width 800, shared=0 | VERIFIED | Dense FFN отсутствует |
| A03 | GQA/QK-Norm before RoPE; untied embeddings | VERIFIED | D04 projections сохраняются |
| A04 | FP32 router weights/logits, no trainable bias, stable ID ties, dropless | VERIFIED: V3.0 | Нет hidden fallback |
| A05 | Единственный будущий configs/shinra_40b_moe.yaml | VERIFIED: V3.0 | Никаких других YAML |
| A06 | GM управляет базовым stack; без установок и torch в requirements | VERIFIED: V3.0 | Actual GM versions UNKNOWN |
| A07 | Source implementation разрешена | VERIFIED: Order 004 | Повторный GO по модулям не требуется; execution запрещён |
| A08 | Все checkpoint tensor payloads — Safetensors | VERIFIED: прямой приказ 25.10.2026 | Metadata JSON; никаких .pt/.pth/pickle или собственного tensor format |
| P01 | Balancing eta=0.001, bound=2, bounded sign update | PROPOSED: D03 | Не production defaults; training требует approval |
| P02 | RoPE theta=1000000, adjacent pairing, identity scaling, FP64 phases | PROPOSED: D04 | Численная frequency policy ещё не явно утверждена |
| P03 | Attention dropout=0; BF16 cache | PROPOSED: D04 | Не подставлять неявно |
| P04 | Curriculum 8K→32K→64K→128K→256K | PROPOSED: D04 | Budgets/mix UNKNOWN |
| P05 | Special IDs 0–31, byte IDs 32–287, learned от 288 | PROPOSED: D05 | Не создавать tokenizer artifact до approval/training |
| P06 | BPE min frequency 2, max token bytes 256, no regex | PROPOSED: D05 | Не менять vocabulary size при нехватке merges |
| P07 | Data schemas/exact dedup, language evidence, cursor | PROPOSED: D06 | Corpus approval отдельно |
| P08 | Training/checkpoint/distributed contracts D07 | PROPOSED | Не обещать готовые collectives/recovery |
| P09 | Schema v1 и config identity D01 | PROPOSED | No hidden defaults |
| P10 | PyYAML 6.0.3 | Выбран для parser при Order 004; release/license documented | requirements.txt, без установки |
| P11 | Normal(0, 1/sqrt(2560)); residual projections scaled by 1/sqrt(2*48); FP32 sampling then storage cast; logical-name SHA-256 seed derivation | PROPOSED: Engineering Order 005 | Candidate only; not an approved config default or evidence of runtime reproducibility |
| U01 | Initialization approval and cross-device RNG guarantees | UNKNOWN | Initialization BLOCKED until explicit recipe fields are approved; bitwise cross-device identity is not claimed |
| U02 | Optimizer, LR/betas/epsilon, batch, clipping, scheduler/token budget | UNKNOWN | Training BLOCKED, без придуманных чисел |
| U03 | Corpus/manifests/tokenizer artifacts | UNKNOWN / NOT APPROVED | Ingestion/pretraining BLOCKED |
| U04 | Runtime GM versions, backend и logical topology | UNKNOWN | Зависимые adapters BLOCKED; GPU planning вне scope |
| U05 | Storage durability/resharding | UNKNOWN | Safetensors source существует; три P1 исправлены статически, runtime и durability не подтверждены |
| U06 | Chat header grammar, FIM sampling/loss policy | UNKNOWN | Соответствующая сериализация/обучение требуют решения |

## Сверка неоднозначностей

Order 004 предоставил требуемое разрешение: исходная реализация ведётся. Запрет исполнения сохраняется.

Фраза «сохранить утверждённый контракт RoPE» не содержит явного принятия численного theta из ранее PROPOSED D04. Theta не изменяется, но и не переименовывается в approved. Аналогично сохранение D05 не доказывает approval предложенных ID. Это не конфликт двух утверждённых чисел, а отсутствующее решение.

D03 говорит об update после успешного optimizer commit; D07 уточняет единый logical transaction: внешне completed step публикуется только после согласования весов/bias/scheduler/cursor. Ошибка после частичного изменения требует recovery. Нет ложной гарантии atomic in-memory optimizer mutation.

D05 exact BPE требует границ content fragments. Data streaming не имеет права менять токенизацию произвольным byte chunking. Short-document packing не подтверждает 256K dependencies.

## Founder review

Документальный комплект представлен для подтверждения. Требуются решения P01–P10 и U01/U06 для соответствующих компонентов; U02–U05 могут оставаться явно UNKNOWN в config с operation gates. Неизвестный корпус не должен блокировать независимую реализацию конфигурации/математики после разрешения.

Архитектурных размерностей не изменено. Ни одно PROPOSED решение не является результатом измерения.

## Re-audit 25.10.2026

[Current findings](AUDIT_REMEDIATION_2026-10-25.md), [initialization candidate](INITIALIZATION_RECIPE.md). Никакие numeric proposals не переведены в executable defaults. Publication/training resume NO-GO сохраняется.
