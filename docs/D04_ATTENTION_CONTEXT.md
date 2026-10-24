# D04 — Attention and Native 256K Context

Дата: 24.10.2026. Contract candidate: shinra.attention.v1.

## VERIFIED — требования

Order 003 фиксирует GQA, Q/K/V/O shapes, head dimension 128, independent per-head QK-Norm до RoPE, scale 1/sqrt(128), causal attention и target context 262144. MLA не внедряется. Остальные детали ниже PROPOSED.

## PROPOSED — projections

Для X[T,2560] после pre-attention RMSNorm:

\[
Q=XW_Q^T,\quad K=XW_K^T,\quad V=XW_V^T
\]

| Weight | Shape [out,in] |
|---|---|
| W_Q | [4096,2560] |
| W_K | [1024,2560] |
| W_V | [1024,2560] |
| W_O | [2560,4096] |

После reshape: Q[T,32,128], K/V[T,8,128]. Query head h использует KV head g(h)=floor(h/4), h=0…31. Bias отсутствуют.

## PROPOSED — QK-Norm

\[
\widehat q_{t,h,j}=\gamma^Q_{h,j}\frac{q_{t,h,j}}{\sqrt{\frac1{128}\sum_{r=0}^{127}q_{t,h,r}^2+10^{-6}}}
\]

Аналогично K. Gamma_Q[32,128], gamma_K[8,128], начальные значения единицы. Mean square и reciprocal sqrt FP32. Beta и дополнительной trainable temperature нет. V не нормализуется. QK-Norm применяется перед RoPE, порядок не переставляется.

CALCULATED: (32+8)×128 = 5120 trainable QK-Norm parameters/layer.

## PROPOSED — окончательное предложение RoPE

| Поле | Значение |
|---|---|
| rope_theta | 1000000 |
| rotary_dim | 128, вся голова |
| Pairing | Соседние координаты (0,1),(2,3),… |
| Position scaling | f(p)=p |
| Dynamic/YaRN/NTK scaling | Не включены |
| Дополнительный attention multiplier | Отсутствует |
| Допустимые позиции | 0…262143 |

Для i=0…63:

\[
\omega_i=\theta^{-2i/128},\quad\phi_{p,i}=p\omega_i
\]
\[
\begin{pmatrix}q'_{2i}\\q'_{2i+1}\end{pmatrix}
=\begin{pmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{pmatrix}
\begin{pmatrix}\widehat q_{2i}\\\widehat q_{2i+1}\end{pmatrix}
\]

Та же операция применяется к нормализованным keys.

VERIFIED — математическое происхождение: [RoFormer, 2104.09864v5](https://arxiv.org/abs/2104.09864v5), просмотрено 24.10.2026. Theta=10^6 — PROPOSED SHINRA, не подтверждённый optimum. Модель с нуля использует одну систему координат во всём curriculum и decode, без смены частот при увеличении текущей длины.

CALCULATED:

\[
\omega_{63}\approx1.24094\times10^{-6},\quad 2\pi/\omega_{63}\approx5\,063\,256
\]
\[
262143\omega_{63}\approx0.325303
\]

Это свойства частот, не доказательство освоения дальних зависимостей. [YaRN, 2309.00071v3](https://arxiv.org/abs/2309.00071v3) исследован как альтернатива; не входит в контракт. Изменение scaling требует новой RoPE version и отдельного решения.

## PROPOSED — precision и positions

Position IDs целочисленные, без промежуточного BF16 округления. Канонический расчёт частот/фаз FP64, sin/cos приводятся к FP32, вращение FP32 с cast в attention dtype. Иной backend численный путь должен быть явно описан и сохранять формулы. Bitwise trig equivalence не предполагается.

Позиции вне диапазона вызывают ошибку. Нет автоматических wraparound, rescaling или sliding window.

## PROPOSED — attention и mask

\[
a_{t,h,u}=\frac{\langle q'_{t,h},k'_{u,g(h)}\rangle}{\sqrt{128}}+M_{t,u}
\]
\[
M_{t,u}=\begin{cases}0&d_t=d_u,\ p_u\le p_t,\ \text{оба токена действительны}\\-\infty&\text{иначе}\end{cases}
\]
\[
P_{t,h,:}=\operatorname{softmax}(a_{t,h,:}),\quad O_{t,h}=\sum_uP_{t,h,u}V_{u,g(h)}
\]
\[
Y_t=\operatorname{concat}_{h=0}^{31}(O_{t,h})W_O^T
\]

d — logical attention segment ID, не просто filename. Документ/диалог — отдельный segment. Разные packed documents не видят друг друга. PAD не query/key. Действительный query имеет как минимум собственный key; полностью masked строки не передаются backend. Softmax и accumulation FP32. Attention dropout=0 — PROPOSED.

Block-diagonal causal mask задаётся boundary metadata; материализация квадратной mask не требуется.

## PROPOSED — boundaries

- BOS имеет position 0; все реальные structural/content tokens увеличивают position на один.
- Независимый документ сбрасывает positions; chat turns одного диалога не сбрасывают.
- FIM использует сериализованный порядок, не исходные offsets кода.
- PAD не продвигает positions. Максимальная длина включает specials.
- Разделение длинного документа на независимые segments разрывает внимание. Каждый получает segment ID и positions от нуля; это фиксируется в manifest и не считается обучением полной длинной зависимости.

## PROPOSED — KV-cache

Логическая форма на слой/sequence: K_cache[S,8,128], V_cache[S,8,128]. Хранятся K после QK-Norm/RoPE, V после projection; committed length; segment identity; position metadata; model/checkpoint identity; RoPE identity; dtype/layout version.

Cache BF16, quantization не включена. Paging/размещение не меняют логический порядок. Нельзя переиспользовать cache с другими weights/RoPE/sequence identity.

## PROPOSED — prefill/decode equivalence

Prefill использует те же positions/mask, что последовательный decode. Chunked prefill использует абсолютную позицию внутри segment, не локальный chunk offset. Decode token получает position=committed_length; его K/V доступны собственному query. Новый length публикуется после успешного завершения всех слоёв. Незавершённое расширение не committed; повтор не дублирует token.

Training/inference attention математически эквивалентны при одинаковых weights, sequence, boundaries, positions/RoPE и отсутствии dropout. Округления и порядок операций могут отличаться; bitwise equivalence не обещается. Training не переиспользует cache между optimizer steps.

## PROPOSED — curriculum

\[
8192\rightarrow32768\rightarrow65536\rightarrow131072\rightarrow262144
\]

Одна модель и fixed RoPE, длины stages не меняют architectural maximum. Допускаются короткие segments на каждом этапе. Переход требует согласованного token budget и реальных документов нужной длины. Packed короткие изолированные документы не заменяют одну 256K dependency. Искусственные повторы для заполнения длины запрещены. Длительности stages/доли sequences UNKNOWN до корпуса. Даже последний stage не доказывает long-context quality.

## PROPOSED — backend interfaces

| Interface | Contract |
|---|---|
| Training/prefill | Q/K/V, boundaries, positions, scale → output, backward context |
| Decode | Новые Q/K/V, committed cache, positions → output, transactional append |
| Capabilities | Dtypes, head dims, GQA ratio, causal/varlen, backward, determinism |
| Unsupported semantics | Явная ошибка |

Backend не уменьшает context, не вводит window, не меняет masks/heads, не вращает cached keys повторно и не подменяет exact attention approximation без решения. Context parallel interface описывает logical shards/global positions, не реальное количество GPU или topology.

## UNKNOWN / BLOCKED

UNKNOWN: качество theta, освоенный effective context, совместимость конкретного backend. BLOCKED: обучение и подтверждение 256K до корпуса, разрешения и фактического исполнения. D04 подготовлен. NOT EXECUTED: model runtime, тесты, обучение, inference. VRAM budgets и GPU planning не выполнялись.
