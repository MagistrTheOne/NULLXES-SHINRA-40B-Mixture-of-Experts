# D06 — Data pipeline

24.10.2026. VERIFIED: corpus UNKNOWN / NOT APPROVED. Никаких источников, URL, sample data или корпусов не добавлено. Далее PROPOSED контракты.

## Records и входной допуск

SourceRecord: source_id, revision, founder_approval_reference, license_identifier, license_text_digest, permitted_use, owner/provenance, content_manifest_identity. Указание строки лицензии не доказывает права: отсутствие разрешения блокирует ingestion.

DocumentRecord: source_id/revision, document_id, content_location из утверждённого manifest, raw_sha256, byte_length, decoding, language_label, language_method/version, document metadata. Document identity отделена от content hash: одинаковый текст может иметь несколько provenance records.

Manifest: schema_version, упорядоченные shard identities/hashes/record counts, source registry identity, transformation policy, tokenizer identity (для tokenized stage), rejected/quarantined counts. Manifest immutable; изменение создаёт новую identity. Streaming cursor не указывает на mutable directory listing.

## Pipeline

Registry approval → verified manifest → ingestion → строгий UTF-8 → content identity → exact deduplication → language metadata → approved tokenizer → segment construction → packing → training stream.

Каждая стадия возвращает record либо typed rejection с исходной identity/reason; нет молчаливого пропуска. Raw bytes сохраняют исходный hash. Text transformation identity отдельно от raw identity. По D05 Unicode/newline normalization не выполняется. Формат container ingestion отделён от содержимого документа; новый формат требует явного adapter, не guessed decoding.

## Deduplication и language

Exact dedup: SHA-256 payload с byte comparison при совпадении digest, deterministic first-record retention по manifest order. Duplicate provenance сохраняется в index. Near-dedup algorithm/threshold UNKNOWN; exact dedup не выдаётся за устранение near duplicates.

Language label enum: ru, en, mixed, other, unknown. Нужны method/version и provenance; неизвестный язык не угадывается по нескольким символам. Automated detector не выбран и не добавляется dependency. Политика допуска unknown/mixed и sampling weights — обязательные operational values, UNKNOWN до решения.

## Segment / PackedBatch

Segment: segment_uid, document identity, byte/token span, token IDs, positions, loss eligibility, structural boundaries. Splitting длинного документа явно фиксируется; нельзя приписывать независимым кускам обучение полной зависимости.

PackedBatch: input_ids[B,S], positions[B,S], segment_ids[B,S], valid_mask[B,S], next_token_loss_mask[B,S-1], ordered token_uids, tokenizer/dataset identities. Выход может быть compact [T] с offsets, если обратим к тем же metadata.

Для пары позиций i→i+1 loss допустим только при valid обеих позициях, одинаковом segment и разрешённой target semantics. PAD, DOC_SEP как target и cross-document BOS исключены; EOS предсказывается. DOC_SEP относится к предыдущему segment, positions D04/D05. Packing учитывает specials и capacity; границы attention не выводятся из текстовых имён tokens.

## Reproducible streaming

Cursor: manifest identity, epoch, shard/document ordinal, fragment/token offset, packing pending state, shuffle algorithm/seed/state, logical partition identity. Read-ahead cursor отдельно от committed cursor. Engine commit подтверждает только использованные записи; abort возвращает к committed state. Перестановка worker arrival не меняет logical order.

Долговечный cursor checkpoint обязан включать незавершённый packing либо ссылки, позволяющие его точно восстановить. Changing logical partition/world configuration при resume не выполняется автоматически: требуется явный reshard policy. Data replication не означает повторное прибавление одного token_uid в routing stats.

## Interfaces

register_source(record) → approved registry entry; ingest(manifest,cursor) → documents + pending cursor; deduplicate(record,index) → retained/duplicate record; tokenize(record,tokenizer_identity) → segments; pack(segments,capacity) → PackedBatch; commit_cursor(transaction_id) → durable stream state. Side effects отсутствуют при import/config parsing.

## UNKNOWN / BLOCKED

Источники, лицензии реального корпуса, manifest, tokenizer artifact, sampling и filtering thresholds UNKNOWN. Ingestion/pretraining BLOCKED без approved manifest. Независимая реализация interfaces после разрешения возможна без корпуса. NOT EXECUTED: data download/generation/ingestion/tokenization.

## Order 010 packing contract — 29.10.2026

`training_sequence_budget(config)` is the single explicit per-source global sequence budget. `DocumentPacker.capacity` uses it, while `context_capacity` stays 262144 and defines stable UID stride. Stream packing includes BOS/EOS/separator overhead and checks capacity before batch materialization. Training checks the complete sequence before CP, then CP owns floor-divided global prediction ranges with real cross-shard next-token targets. Oversized whole documents are rejected with source/document/ordinal/length diagnostics; no segmented continuation is inferred from this proposal. Abort restores the committed document cursor. No datasets or sample documents were added.
