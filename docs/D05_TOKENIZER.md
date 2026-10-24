# D05 — Native Byte-Level BPE Tokenizer

Дата: 24.10.2026. Contract candidate: shinra.bytebpe.v1.

## VERIFIED — требования

Order 003: собственный tokenizer, vocabulary 131072, RU/EN, полное byte coverage. Корпус NOT SELECTED. Чужие vocabulary/merges не используются. Дополнительные правила ниже PROPOSED.

## PROPOSED — vocabulary allocation

| IDs | Назначение | Slots |
|---|---|---:|
| 0–31 | Structural/reserved specials | 32 |
| 32–287 | Bytes 0–255 | 256 |
| 288–131071 | Learned BPE byte sequences | 130784 |

CALCULATED: 32+256+130784=131072. Byte b имеет token_id=32+b. Отображение <0xXX> лишь диагностическое, не payload.

## PROPOSED — конкретные special IDs

| ID | Name | Семантика |
|---:|---|---|
| 0 | PAD | Исключён из attention/loss |
| 1 | BOS | Начало независимого segment |
| 2 | EOS | Конец document/dialogue |
| 3 | DOC_SEP | Packing separator |
| 4 | ROLE_SYSTEM | System message |
| 5 | ROLE_USER | User message |
| 6 | ROLE_ASSISTANT | Assistant message |
| 7 | ROLE_TOOL | Tool message |
| 8 | HEADER_END | Header/body boundary |
| 9 | MESSAGE_END | Конец message |
| 10 | TURN_END | Конец assistant turn |
| 11 | FIM_PREFIX | Prefix marker |
| 12 | FIM_SUFFIX | Suffix marker |
| 13 | FIM_MIDDLE | Middle marker |
| 14 | FIM_END | Конец FIM representation |
| 15–31 | RESERVED_15…RESERVED_31 | Не назначены, запрещены в обычном input |

UNK отсутствует: все bytes представимы. Reserved slots заявлены структурно, не являются фиктивными learned pieces. Их embedding/head rows входят в D02.

## PROPOSED — UTF-8/Unicode

- encode_text принимает Unicode scalar values и кодирует UTF-8; unpaired surrogates вызывают ошибку.
- Нет NFC/NFD/NFKC/NFKD, case folding, transliteration или homoglyph replacement.
- encode_bytes обратимо принимает любые bytes; это не разрешение вводить binary data в training corpus.
- Ill-formed UTF-8 при text ingestion отклоняется/quarantine, без скрытого исправления.

VERIFIED — источник правил корректности UTF-8: [Unicode 17.0.0, Chapter 3](https://www.unicode.org/versions/Unicode17.0.0/core-spec/chapter-3/), просмотрен 24.10.2026. Конкретная политика SHINRA PROPOSED.

## PROPOSED — whitespace/code

Сохраняются spaces/counts, tabs, LF/CRLF/CR, NBSP/Unicode spaces, leading/trailing whitespace, входной BOM и NUL в byte API. Нет prefix space, trim или newline conversion. Код использует тот же BPE: indentation/operators/Unicode identifiers не переписываются. Language-specific tokenizer отсутствует.

## PROPOSED — BPE training contract

Initial sequence — byte IDs. Regex pre-tokenizer отсутствует. Merges только внутри одного обычного text/byte fragment, не через structural boundaries или documents. Maximum ordinary token length=256 bytes; BPE dropout отсутствует.

На будущем согласованном корпусе:

1. Подсчитать adjacent pair frequencies в текущей сегментации.
2. Исключить результирующие payload длиннее 256 bytes.
3. Исключить payload, уже имеющие обычный token ID.
4. Выбрать максимальную частоту; tie-break — минимальная пара (left_id,right_id) лексикографически.
5. Minimum pair frequency=2.
6. Назначить следующий ID от 288.
7. Применить merge слева направо без перекрытий.
8. Повторять до 130784 learned slots.

Если допустимых merges недостаточно — admission BLOCKED. Нельзя добивать vocabulary искусственными pieces, уменьшать размер или заявлять полный обученный словарь.

VERIFIED — происхождение BPE subword method: [Sennrich et al., 1508.07909v5](https://arxiv.org/abs/1508.07909v5), просмотрено 24.10.2026. Byte alphabet, IDs и детали выше — собственное предложение SHINRA.

## PROPOSED — deterministic encoding

Преобразовать input в byte IDs; найти применимые adjacent pairs; выбрать minimum merge rank; при нескольких вхождениях одного rank — крайнее левое; обновить соседние пары; повторить до отсутствия merges. Rank однозначен порядком обучения. Batch boundaries не меняют результат.

Нельзя независимо токенизировать произвольные chunks одного документа и просто конкатенировать IDs: merge может пересечь chunk boundary. Канонический API финализирует законченный content fragment. Streaming optimization обязана сохранить ту же сегментацию.

## PROPOSED — decoding

decode_bytes конкатенирует ordinary byte payload. decode_text сначала восстанавливает bytes, затем strict UTF-8. Отдельный token может содержать неполный UTF-8 character. Streaming decoder удерживает незавершённую последовательность. Некорректный завершённый UTF-8 даёт ошибку с доступом к bytes, не скрытую замену на U+FFFD.

Special tokens не имеют ordinary byte payload. decode_structured возвращает content fragments и structural events. Пропуск specials/печатные имена — явно выбранная presentation policy.

CALCULATED — алгоритмический инвариант, не выполненный тест:

\[
\operatorname{decode\_bytes}(\operatorname{encode\_bytes}(b))=b
\]

Каждый merge заменяет два payload их конкатенацией.

## PROPOSED — trusted specials и chat

Печатное имя special внутри обычного user text не преобразуется в structural ID. Specials добавляет доверенный serializer из типизированных полей.

Grammar: BOS; role token → header fields → HEADER_END → body. System/user/tool message заканчивается MESSAGE_END. Assistant intermediate message может иметь MESSAGE_END; завершённый assistant turn — TURN_END. Полный dialogue заканчивается EOS.

Header fields требуют отдельной versioned grammar (ещё не утверждена). Tool names не создают новые IDs. Chat tokens не означают обученное chat behavior или готовый tool execution subsystem.

## PROPOSED — FIM

\[
\text{BOS},\text{FIM\_PREFIX},P,\text{FIM\_SUFFIX},S,\text{FIM\_MIDDLE},M,\text{FIM\_END},\text{EOS}
\]

Prefix/suffix/middle fragments токенизируются отдельно. Merges не пересекают markers. Все части одного attention segment; position IDs соответствуют serialized order, не original offsets. FIM frequency, cuts и loss policy UNKNOWN. FIM данные не создаются автоматически.

## PROPOSED — sequence packing

Обычный record: BOS, encode(document), EOS. Между records вставляется DOC_SEP.

- DOC_SEP принадлежит предыдущему segment и получает position после EOS.
- Новый BOS начинает новый segment/position 0.
- DOC_SEP не target loss; переход DOC_SEP→следующий BOS также исключён.
- Предсказание EOS после content входит в causal LM objective.
- PAD исключён из attention/loss/routing.
- Length limit включает весь structural overhead.
- Если separator не помещается, новая запись начинается в другом packed container.
- Mask строится по explicit segment metadata, не textual markers или одному EOS.

## PROPOSED — serialization/versioning

Логические артефакты будущего bundle (сейчас не созданы):

| Artifact | Содержание |
|---|---|
| Manifest | Format/algorithm version, vocab size, hashes |
| Vocabulary | ID→special descriptor или bytes в hex |
| Merges | Rank,left ID,right ID,output ID |
| Special registry | ID,name,semantics,allowed contexts |
| Training provenance | Approved corpus hash, data order, trainer parameters |
| Statistics | Реально измеренные distributions; до обучения отсутствуют |

Serialization UTF-8, byte payload hex. Vocabulary упорядочен по ID, merges по rank. SHA-256 считается для точных bytes artifacts. Identity включает все semantic artifacts, model checkpoint хранит точный tokenizer identity. Изменение IDs/merges/normalization/segment rules создаёт несовместимую semantic version. Нельзя обновлять под старым identity.

Static admission: уникальность IDs/payload, полный byte alphabet, корректные merge references, возрастающий порядок создания, точное число slots. Это контракт, не созданные test cases.

## PROPOSED — training interface/statistics

Input: approved corpus manifest, document IDs/content hashes, licenses/provenance, deterministic order, explicit RU/EN sampling policy, BPE parameters.

Output: vocabulary/merges, bundle manifest, фактические RU/EN statistics, special/ordinary frequencies, bytes/token, tokens/document, длинные/повторяющиеся token sequences, признаки недостаточности корпуса.

RU/EN не получают произвольные fixed vocabulary quotas. Learned pieces определяет согласованный корпус. Byte representability не означает одинаковой эффективности RU/EN segmentation.

## UNKNOWN / BLOCKED

UNKNOWN: корпус, distributions, достижимость learned vocabulary, compression quality, FIM/chat training policy и окончательная header grammar. BLOCKED: tokenizer training, создание vocabulary/merges и заявления о качестве до утверждения корпуса и разрешения выполнения. D05 подготовлен как предложение. NOT EXECUTED: corpus selection/download/generation, tokenizer training, tests.
