# D07 — Pretraining, distributed state and checkpoints

24.10.2026. VERIFIED: pretraining с нуля; execution запрещён. Далее PROPOSED, operational numbers не назначаются.

## Objective и accumulation

\[
\ell_i=-z_{i,y_i}+\log\sum_{v=0}^{131071}\exp z_{i,v},\qquad
\mathcal L=\frac{\sum_i m_i\ell_i}{\sum_i m_i}
\]

Logsumexp стабилен, accumulation FP32. Shift labels ровно на одну позицию внутри segment; masks D06. Суммируется loss только настоящих target tokens, не среднее microbatch means. N=Σm по уникальным tokens глобального optimizer step. N=0 блокирует update. Для distributed sum gradients масштаб 1/N; если API усредняет по DP group size R, локальный summed loss масштабируется R/N. TP/CP copies не считаются новыми loss targets. Chunked LM head/loss должен сохранять полный vocabulary denominator, без sampled softmax.

## Optimizer/scheduler — интерфейс

Optimizer integration получает явные algorithm, LR, betas/epsilon/weight_decay (если нужны algorithm), parameter-group policy и precision states. Ни optimizer class, ни коэффициенты не выбираются defaults. Scheduler получает algorithm, единицу progress (tokens или successful steps), warmup/decay/total budget. Advance только после commit; failed attempts не тратят committed token budget.

Gradient accumulation задаётся явно; сохраняется global normalization. Clipping — global norm уникальных trainable gradients после синхронизации и unscale, перед update; max_norm и norm_type обязательны. Не clip-ить каждый microbatch отдельно. Gradients всех experts, в том числе отсутствовавших в routing, имеют явную zero/None policy для выбранного optimizer.

## Precision/recomputation

BF16 residual/expert path — baseline candidate; router FP32 обязателен. Norm statistics, selected softmax и combination FP32. Нет глобального model.to(bfloat16), которое молча понизит router. Optimizer/master state dtypes объявляются отдельно. FP8 backend не выбран. Loss scaling, если выбран, включается явно, а overflow abort-ит transaction без продвижения scheduler/cursor/bias.

Recomputation сохраняет routing assignment IDs, frozen balancing version и необходимые RNG states; повторный forward не обновляет telemetry counts/bias. Non-reentrant checkpoint API рассматривается в D09, но конкретная реализация ожидает подтверждения стека.

## State transitions

Committed(s) → open attempt → accumulation → synchronize counts/gradients → finite validation → global clipping → prepare → optimizer update → balancing update → scheduler/cursor commit → committed(s+1).

Commit логический, не обещание атомарного физического изменения всех tensors. При ошибке до mutation pending state отбрасывается; при частичной mutation требуется recovery целостного checkpoint. Если durable checkpoint старее RAM commit, повторяется весь участок после durable cursor. D03 load accounting согласуется с этими границами.

## Distributed contracts

Logical topology предоставляется извне. Process groups, ranks и ownership обязательны, никакого universal world_size=1 fallback.

| Компонент | Семантика |
|---|---|
| DP | Уникальные data partitions; явное sum/mean gradient convention |
| TP | Shard spec каждой matrix: logical shape, axis, offsets; восстановление полных математических операций |
| EP | Owner (layer,expert), dispatch counts/offsets, real all-to-all; reverse exchange в backward |
| CP | Partition sequence positions; полная causal/document mask сохраняется |
| Communication | Явные groups, dtype, counts, operation order и timeout |

Группы могут пересекаться только по валидированному mapping. Autograd через collectives не предполагается автоматически: backward должен иметь математически обратный communication plan. Повторы token_uid от replication исключены из статистики. Все ranks участвуют в согласованной последовательности collectives, включая пустые local shards.

Для CP предлагается exact blockwise attention: для объединяемых key blocks держать maximum m, normalizer l и numerator o. При merge двух состояний m'=max(m1,m2), l'=exp(m1-m')l1+exp(m2-m')l2, o'=exp(m1-m')o1+exp(m2-m')o2; output=o'/l'. Полностью masked block не вычисляет -Inf-(-Inf), а пропускается. Это математический контракт, не утверждение готового distributed kernel/backward.

Failure coordination не гарантируется живым collective после потери rank. External launcher/supervisor должен завершить повреждённую group и организовать restart из durable state. Конкретный launcher UNKNOWN. Незавершённый distributed backend блокируется, а не симулируется.

## Checkpoint format — Safetensors (распоряжение NULLXES 25.10.2026)

Manifest содержит schema/version, transaction ID, successful step, config identities, tokenizer/dataset identities, tensor shard names/shapes/dtypes/offsets/hashes, optimizer/scheduler states, router buffers, committed cursor, per-logical-worker RNG, ownership map, precision policy и source revision.

Формат всех сохраняемых tensors — Safetensors: model weights, tensor optimizer state, router buffers и tensor RNG state. Никаких альтернативных .pt/.pth/pickle весов или самодельного tensor container. Прежнее shinra.checkpoint.v1 больше не используется как значение format. Служебный manifest содержит JSON metadata, ссылки на .safetensors shards и hashes; это не альтернативный формат весов. Scalars/структура optimizer/scheduler/cursor сериализуются как данные, не executable objects. Целостность: SHA-256 каждого shard и manifest. Фактические файлы checkpoint создавать сейчас запрещено.

Publication: выбрать уникальный transaction ID → snapshot committed boundary → записать все shards во временное transaction namespace → flush/durability по backend contract → собрать acknowledgements и hashes → manifest → publish COMMITTED marker последним. Reader принимает только полную committed transaction с проверенными identities/hashes. LATEST — подсказка, не доказательство целостности.

Atomic rename допустим лишь если storage backend гарантирует требуемую атомарность и durability на том же filesystem. Для object store нужен отдельный conditional publication contract; не изображать rename. Partial files не удаляются автоматически и не считаются checkpoint. Во время snapshot веса не меняются либо используется явно согласованная immutable snapshot strategy.

Recovery: validate manifest/schema → identities/shards/ownership → восстановить модель, optimizer/scheduler/bias/cursor/RNG → подтвердить одну transaction на всех участниках → resume. Config drift, tokenizer mismatch и silent data skipping запрещены. Resharding между топологиями отдельная явно поддержанная операция, пока UNKNOWN.

## Telemetry

Только фактические loss numerator/denominator, consumed/committed tokens, attempts/steps, gradient norm, expert counts, bias saturation, failure reasons, checkpoint transaction IDs и реально измеренное время. События original forward/recompute различаются. Throughput/latency отсутствуют без измерения; нет fake progress.

## UNKNOWN / BLOCKED

Optimizer/schedule/coefficients, batch/accumulation, clipping, token budget, initialization, corpus, GM API version, chosen backends, storage durability UNKNOWN. Соответствующие операции BLOCKED, архитектурные interfaces не подменяются defaults. Runtime correctness не подтверждена. NOT EXECUTED: training, model initialization, checkpoint save/load, collectives, tests.
