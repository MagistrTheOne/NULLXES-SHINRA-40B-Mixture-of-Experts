# D01 — SHINRA-40B-MoE architecture

24.10.2026. VERIFIED требования: Engineering Order 003 и V3.0. Новые системные решения PROPOSED; создание документа не означает их утверждения.

## Математическая модель

Token IDs [B,S] → E[131072,2560] → H_0[B,S,2560]. Для l=0…47:

\[
A_l=H_l+\operatorname{GQA}_l(\operatorname{RMSNorm}_{a,l}(H_l)),\qquad
H_{l+1}=A_l+\operatorname{MoE}_l(\operatorname{RMSNorm}_{m,l}(A_l))
\]
\[
\operatorname{logits}=\operatorname{RMSNorm}_f(H_{48})W_{out}^{T}
\]

W_out[131072,2560] не связан с E. RMSNorm(x)=gamma*x/sqrt(mean(x²)+1e-6). Bias/beta нет. Формы packed execution [T,...] эквивалентны только при сохранении segment metadata. MoE имеется в каждом блоке; shared experts=0. Dense FFN, скрытых альтернатив и чужих весов нет.

| Invariant | Value |
|---|---|
| Hidden/layers | 2560 / 48 |
| Q/KV heads, head dim | 32 / 8 / 128 |
| Q/K/V/O | 2560→4096; 2560→1024; 2560→1024; 4096→2560 |
| Experts / selected / intermediate | 128 / 8 / 800 |
| Vocabulary / context target | 131072 / 262144 |
| Total / conventional active | 39694338560 / 4304898560 |

Расчёт: [D02](D02_PARAMETERS.md). Routing: [D03](D03_ROUTING.md). Attention: [D04](D04_ATTENTION_CONTEXT.md). Tokenizer: [D05](D05_TOKENIZER.md).

## Configuration contract — PROPOSED

Единственный будущий YAML: configs/shinra_40b_moe.yaml. Schema version — явное поле; версия 1 предложена. Обязательные top-level sections: schema_version, model, attention, rope, moe, router, tokenizer, data, training, distributed, checkpoint, inference. Только model хранит фундаментальные размерности; остальные sections ссылаются на них, не дублируют численные источники истины.

| Section | Ответственность |
|---|---|
| model | Все поля approved baseline V3.0, initialization policy и её approval |
| attention | QK order, normalization, scaling rule, mask и выбранный backend |
| rope | theta, pairing, frequency dtype, scaling policy, approval |
| moe | dispatch/combine semantics, overflow/chunk policy |
| router | FP32 policy, tie-break, balancing algorithm, eta/bound/approval |
| tokenizer | Contract identity, special registry, merge policy, artifact identity |
| data | Approved source manifest, identity, language/dedup policy |
| training | Optimizer, scheduler, accumulation, clipping, token budget, precision |
| distributed | Explicit logical groups/ownership, backend, failure policy |
| checkpoint | Format, destination, durability contract, recovery policy |
| inference | Sampling, stops, output length, checkpoint/tokenizer identities |

Schema должна отклонять неизвестные ключи на каждом уровне, duplicate mapping keys, aliases/merge keys, custom tags, nonfinite numbers и неверные типы (bool не integer). Один документ YAML; никаких скрытых env overrides и profiles. Неподтверждённые operational values — null с явной диагностикой операции, не подставленные defaults. Предложения из D03–D05 не становятся executable values без approval.

Validation имеет два уровня: architecture validation разрешена при неизвестном corpus; operation validation отдельно требует initialization/inference/training prerequisites. Ошибка содержит путь поля и причину. Проверяются shapes, D02 algebra, head grouping, отсутствие shared experts, untied weights и позиционные границы. Значения не исправляются автоматически.

Config identity — SHA-256 канонической JSON-сериализации проверенных config fields (sorted keys, UTF-8, без NaN/Infinity). Отдельные architecture и operation identities позволяют документировать изменение checkpoint destination без молчаливого изменения модели; точный whitelist operational changes обязателен при resume.

## Initialization — PROPOSED / UNKNOWN

Только random initialization собственных весов, без I/O при импорте. Initialization требует explicit seed, distribution, scale и residual scaling policy. Числа UNKNOWN, операция BLOCKED до решения. Gamma=1 и balancing b=0 предложены D03/D04. PAD embedding не замораживается автоматически: такая заморозка изменила бы семантику trainable total D02.

Logical parameter names и expert IDs стабильны независимо от sharding. RNG streams привязаны к logical tensor identity; алгоритм и версия генератора входят в checkpoint. Layout-independent bitwise initialization не заявляется до отдельного контракта и исполнения.

## Inference — PROPOSED

Generation request содержит tokenized prompt, model/tokenizer identities, sampling parameters, stop IDs, max_new_tokens и explicit RNG state. Перед prefill проверяется prompt length; новое generation не выходит за 262144. Никакого truncation по умолчанию. На context limit возвращается отдельная причина остановки.

Prefill/decode/cache следуют D04. Предлагаются greedy (tie по меньшему token ID) и categorical sampling: temperature>0, явно заданные top-k/top-p (включая явно выключенные фильтры), stable сортировка, нормализация после фильтрации. Неизвестные values блокируют sampling, NaN/Inf вызывают ошибку. EOS/stop token принадлежит raw output и передаётся structured decoder; UI stripping — отдельная policy. PAD и reserved IDs маскируются только по явно принятой generation policy, не изменяя weights или vocabulary.

Остановка по token IDs, длине и cancellation; byte stop strings требуют отдельного согласованного контракта и сейчас не заявлены. Cache append транзакционный по всем слоям. Abort не публикует частично добавленный token. Inference не обновляет balancing state. Локальный server не создаётся.

## UNKNOWN / BLOCKED

Initialization recipe, окончательное approval RoPE/specials, конкретный attention backend, trained artifacts и runtime UNKNOWN. 256K — target. Реализация ожидает подтверждения после документальной сверки согласно V3.0 §19.
