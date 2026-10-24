# D02 — Parameter accounting

Дата: 24.10.2026. Статус: математический аудит принят Engineering Order 003.

## VERIFIED — зафиксированные размерности

| Обозначение | Значение |
|---|---:|
| Vocabulary V | 131072 |
| Hidden D | 2560 |
| Layers L | 48 |
| Query heads Q | 32 |
| KV heads K | 8 |
| Head dimension H | 128 |
| Experts E | 128 |
| Selected experts k | 8 |
| Expert intermediate F | 800 |
| Shared experts | 0 |
| Target context | 262144 |

Untied embeddings. Все блоки содержат MoE. Attention, expert и router projections не имеют trainable bias. Две pre-norm RMSNorm на блок, independent per-head RMS QK-Norm и одна final RMSNorm. Дополнительные обучаемые температуры отсутствуют в принятом расчёте.

## CALCULATED — размерности

Q projection: 2560 → 4096, K/V: 2560 → 1024 каждая, O: 4096 → 2560. Расширенная Q projection намеренна: QH = 4096, не 2560. Четыре query-head соответствуют одной KV-head.

Матрицы записаны в формате [out_features, in_features].

| Матрица | Форма | Параметры |
|---|---|---:|
| Input embedding | [131072,2560] | 335544320 |
| LM head | [131072,2560] | 335544320 |
| W_Q одного слоя | [4096,2560] | 10485760 |
| W_K одного слоя | [1024,2560] | 2621440 |
| W_V одного слоя | [1024,2560] | 2621440 |
| W_O одного слоя | [2560,4096] | 10485760 |
| Router одного слоя | [128,2560] | 327680 |
| Expert gate | [800,2560] | 2048000 |
| Expert up | [800,2560] | 2048000 |
| Expert down | [2560,800] | 2048000 |

## CALCULATED — формулы total

\[
P_{attention/layer}=2DQH+2DKH=26\,214\,400
\]
\[
P_{expert}=3DF=6\,144\,000
\]
\[
P_{norm}=L[2D+(Q+K)H]+D=494\,080
\]
\[
P_{total}=2VD+L[2DQH+2DKH+E(3DF)+DE+2D+(Q+K)H]+D
\]

| Компонент | Формула | Параметры |
|---|---|---:|
| Input embedding | VD | 335544320 |
| Output head | VD | 335544320 |
| Attention всех слоёв | L(2DQH+2DKH) | 1258291200 |
| Routed experts | LE(3DF) | 37748736000 |
| Routers | LDE | 15728640 |
| Pre-attention/pre-MoE norms | L(2D) | 245760 |
| QK-Norm | L(Q+K)H | 245760 |
| Final norm | D | 2560 |
| **Total** | Сумма | **39694338560** |

В предварительной V1.1 отсутствовали 2560 параметров final RMSNorm. Исправленный расчёт принят основателем.

## CALCULATED — разграничение active/accessed/arithmetic

\[
P_{nonexpert}=P_{total}-LE(3DF)=1\,945\,602\,560
\]
\[
P_{active\ experts}=Lk(3DF)=2\,359\,296\,000
\]
\[
P_{active\ modules}=P_{nonexpert}+P_{active\ experts}=4\,304\,898\,560
\]

| Определение | Значение |
|---|---:|
| Логически хранимые trainable parameters одной копии | 39694338560 |
| Все expert parameters | 37748736000 |
| Выбранные expert parameters на токен | 2359296000 |
| Non-expert parameters | 1945602560 |
| Active с полными embedding/head таблицами | 4304898560 |
| Логически используемые веса с одной input embedding row | 3969356800 |
| Веса арифметики без input embedding lookup | 3969354240 |

Последние строки: P_active_modules − VD + D и P_active_modules − VD соответственно. Предполагается вычисление всех vocabulary logits; input embedding — lookup одной строки. Это не FLOPs и не измеренный memory traffic. Все router weights участвуют в вычислении 128 logits; выбор восьми ограничивает experts, а не исходную router projection.

Physical storage зависит от репликации, sharding, padding и служебных buffers — UNKNOWN. Балансирующие buffers не trainable: 48×128 = 6144 значений и не входят в total. Кэширование/батчинг не меняют число логических параметров.

## PROPOSED / UNKNOWN / BLOCKED

PROPOSED: детали routing, attention и tokenizer описаны отдельно, не утверждаются самим D02. UNKNOWN: runtime, фактическая раскладка хранения, качество обучения. BLOCKED: любые утверждения о выполненной модели без отдельного разрешённого исполнения.

NOT EXECUTED: model initialization, training, inference, тесты. Это аналитический аудит, не результат исполнения модели.
