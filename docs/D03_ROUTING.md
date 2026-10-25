# D03 — Complete MoE Routing Specification

Дата: 24.10.2026. Contract candidate: shinra.routing.v1.

## VERIFIED — требования

Engineering Order 003: 48 MoE blocks; 128 experts/block; Top-8; no shared experts; FFN width 800; FP32 router logits; no trainable router bias; selection использует необучаемый balancing bias, combination — исходные выбранные logits. Все дополнительные правила ниже PROPOSED.

## PROPOSED — формы и precision

Для одного слоя T — число непустых токенов вызова. PAD удаляется до routing, но токены без собственного loss, участвующие в контексте, сохраняются.

| Объект | Форма | Назначение |
|---|---|---|
| X | [T,2560] | Выход pre-MoE RMSNorm |
| W_r | [128,2560] | Trainable FP32 router matrix |
| b | [128] | Non-trainable FP32 balancing buffer |
| Z,S | [T,128] | Raw/selection logits |
| I | [T,8] | Expert IDs |
| P | [T,8] | FP32 combination weights |
| Y | [T,2560] | MoE output до residual |

Router GEMM: FP32 input conversion, weights и accumulation. TF32/FP8 не подставляются автоматически.

## PROPOSED — порядок выбора и softmax

\[
z_{t,e}=\sum_{j=0}^{2559}\operatorname{FP32}(x_{t,j})W_{r,e,j},\qquad s_{t,e}=z_{t,e}+b_e
\]

Top-8 выбирает восемь различных expert IDs по ключу (-s, expert_id): score по убыванию, при точном равенстве FP32 меньший ID первым. Нет epsilon для почти равных scores.

Для выбранных I[t,a], a=0…7:

\[
m_t=\max_a z_{t,I_{t,a}},\quad u_{t,a}=\exp(z_{t,I_{t,a}}-m_t),\quad p_{t,a}=u_{t,a}/\sum_{c=0}^{7}u_{t,c}
\]

Все операции softmax FP32. Balancing bias не входит в combination weights. Полный softmax по 128 experts не требуется. После нормализации пары (expert_id, weight) упорядочиваются по expert_id для dispatch/combine, сохраняя соответствие.

## PROPOSED — NaN/Inf

Fail-closed: NaN/Inf в X, W_r, b, Z, S, expert outputs, combined output или градиентах вызывает ошибку затронутой операции/training attempt. Значения не заменяются нулями, токены не удаляются, experts не переключаются. Distributed ошибка abort-ит всю transaction до подтверждённого commit. Это не гарантия восстановления от произвольного аппаратного сбоя.

Underflow отдельных exp в ноль допустим: максимальный logit даёт exp(0)=1. Нечисловая/нулевая сумма softmax запрещена.

## PROPOSED — expert computation и storage

\[
G_e=X_eW_{g,e}^{T},\quad U_e=X_eW_{u,e}^{T},\quad A_e=\operatorname{SiLU}(G_e)\odot U_e,\quad O_e=A_eW_{d,e}^{T}
\]

W_g и W_u: [800,2560], W_d: [2560,800], bias отсутствуют. Логическое хранение всех experts добавляет ведущую ось [128,...]. Kernel packing обратим и не добавляет обучаемые измерения. Инициализация с нуля; конкретное распределение и scale expert weights не утверждены этим документом.

CALCULATED: 3×2560×800 = 6144000 параметров на expert.

## PROPOSED — стабильный dispatch/combine

Assignment record: layer_id, attempt_id, microbatch_id, token_uid, expert_id, source_owner, source_row, combine_weight, assignment_ordinal. token_uid определяется логической позицией training stream, не временем доставки сообщения.

Внутри dispatch-domain порядок (expert_id, token_uid). Expert output возвращается с исходной assignment identity. Сетевой порядок завершения не определяет combine.

\[
y_t=\sum_{e\in I_t}p_{t,e}O_{t,e}
\]

Умножение weights и сумма FP32 в порядке возрастания expert ID; затем cast в residual dtype. Случайный порядок atomic summation не допускается в детерминированном режиме. Bitwise-совпадение разных backend/hardware не обещается.

## PROPOSED — dropless overflow

Capacity C_e≥1 ограничивает рабочую порцию, а не число принятых assignments. Для n_e назначений число порций ceil(n_e/C_e). Все assignments исполняются тем же expert в стабильном порядке.

Запрещены dropping, повторный Top-8 при перегрузке, переназначение, уменьшение Top-8, другой FFN, изменение bias посреди attempt. Transport backpressure приостанавливает отправку. Невозможность выполнить минимальную порцию означает явный abort до подтверждения optimizer step. Размер порции не меняет математическую семантику; округления GEMM могут зависеть от него.

## PROPOSED — backward

Top-8 дискретен: градиента через сортировку и b нет. Пусть g_t = ∂L/∂y_t и a_(t,e)=〈g_t,O_(t,e)〉.

\[
\frac{\partial\mathcal L}{\partial O_{t,e}}=p_{t,e}g_t
\]
\[
\frac{\partial\mathcal L}{\partial z_{t,e}}=p_{t,e}\left(a_{t,e}-\sum_{j\in I_t}p_{t,j}a_{t,j}\right),\quad e\in I_t
\]

Для невыбранных experts производная данного routing path равна нулю. Input gradient суммирует expert paths и router/combination path. Expert weights получают градиенты всех назначенных токенов. Нулевой objective gradient не исключает изменения веса от optimizer momentum или weight decay — это отдельный optimizer contract.

## PROPOSED — ownership и load statistics

Внешний OwnershipMap определяет владельца (layer_id, expert_id) внутри execution replica. Реплика параметров не новый expert. GPU counts/topology здесь не назначаются.

\[
n_e=\sum_{t\in\mathcal B}\mathbf1[e\in I_t]
\]

B — все уникальные непустые токены microbatches одной optimizer transaction, включая data-parallel части. Counts integer64, с контролем переполнения. TP-копии не дублируют counts; context shards учитывают уникальные токены; recomputation не учитывается. Один назначенный contributor учитывает каждое событие. После accumulation counts согласуются между replica routers. Локально пустой shard допустим, глобально пустой optimizer step не выполняется.

CALCULATED:

\[
\sum_e n_e=8N,\qquad\bar n=8N/128=N/16
\]

## PROPOSED — balancing update

Initial b=0; eta_b=0.001; bound B=2.0; update ровно один раз после успешного optimizer commit; без EMA и дополнительного balancing loss.

\[
d_e=\begin{cases}+1&128n_e<8N\\0&128n_e=8N\\-1&128n_e>8N\end{cases}
\]
\[
b_e^{s+1}=\operatorname{clip}(b_e^s+\eta_b d_e,-B,B)
\]

Сравнение integer, update FP32. Недогруженный expert получает повышение selection score. Saturation вызывает telemetry event, но не изменение коэффициентов. Баланс не гарантируется.

VERIFIED — происхождение идеи: [Auxiliary-Loss-Free Load Balancing, 2408.15664v1](https://arxiv.org/abs/2408.15664v1), первичный источник просмотрен 24.10.2026. Конкретные коэффициенты и bounded update здесь — PROPOSED SHINRA, не доказанный optimum.

## PROPOSED — state machine

| State | Действие |
|---|---|
| COMMITTED(s) | Согласованы weights, optimizer, bias, scheduler, data cursor |
| OPEN(s,attempt) | Frozen bias version; pending counts очищены |
| ACCUMULATING | Original forward добавляет counts; backward — gradients |
| PREPARED | Gradient/count synchronization и проверки конечности завершены |
| COMMITTING | Optimizer update и подготовка новой версии balancing |
| COMMITTED(s+1) | Совместная публикация нового состояния |
| ABORTED | Попытка не считается завершённым optimizer step |

Activation recomputation использует сохранённые expert IDs и frozen b version. Logits/selected softmax пересчитываются с градиентами; counts/state не меняются.

Ошибка до изменения weights: pending counts/gradients отбрасываются, committed bias/scheduler/cursor не продвигаются. После частичного изменения weights/optimizer продолжать запрещено: требуется восстановление целостного committed checkpoint. Если durable checkpoint старше, данные после него повторяются. Уменьшение step_id не является откатом.

## PROPOSED — interfaces

| Interface | Input | Output / obligation |
|---|---|---|
| route | X,W_r,b,token IDs,state version | I,P,assignment plan |
| dispatch | plan,X,ownership | Стабильные expert inputs |
| expert_compute | expert ID,inputs,weights | Outputs той же token axis |
| combine | outputs,original plan | Y; ровно 8 assignments/token |
| accumulate_load | Original-forward events | Counts без повторного учёта |
| prepare_step | Counts,transaction context | Согласованная статистика |
| commit_balance | Successful commit token,counts | Новая b version не более 1 раза |
| abort_attempt | attempt ID | Удаление pending state |

## PROPOSED — checkpoint

На router: W_r и optimizer state; FP32 b; eta_b/B/algorithm version; successful step ID; balancing version; tie-breaking/normalization policy; logical expert IDs; общий checkpoint transaction ID. Durable checkpoint только на committed optimizer boundaries; pending accumulation не выдаётся за завершённый step.

CALCULATED: 48×128=6144 non-trainable balancing values, исключены из D02 trainable total.

## UNKNOWN / BLOCKED

UNKNOWN: оптимальность коэффициентов, реальная нагрузка, устойчивость обучения, окончательная initialization policy. Исходная реализация разрешена Order 004, local MoE primitives существуют. BLOCKED: runtime и интеграция с отсутствующим training/distributed engine. NOT EXECUTED: модель, тесты, обучение, inference.
