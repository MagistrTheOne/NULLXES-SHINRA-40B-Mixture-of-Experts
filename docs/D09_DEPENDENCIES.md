# D09 — Dependency register

Обновлено 25.10.2026. Исходники используют yaml, torch, Safetensors и stdlib. requirements.txt фиксирует PyYAML==6.0.3 и safetensors==0.6.2; torch предоставляется GM. Пакеты не устанавливались. Совместимость конкретной версии GM не подтверждена.

| Component | Version evidence | License/source | Admission |
|---|---|---|---|
| PyTorch, managed by GM | Документация 2.14 просмотрена; фактическая версия GM и минимальная поддерживаемая версия SHINRA UNKNOWN | [Official LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE), BSD-style terms и third-party notices | Базовая потребность V3.0; compatibility UNKNOWN; не добавлять torch в requirements |
| PyYAML | 6.0.3 опубликован, Python >=3.8 по metadata | [Official project release](https://pypi.org/project/PyYAML/6.0.3/), MIT | Прямой parser dependency, выбран в исходниках; не установлен |
| Python stdlib | Версия GM UNKNOWN | Поставляется с интерпретатором | json/hashlib/pathlib/dataclasses/enum/typing/codecs и др. не помещаются в requirements |
| setuptools | 84.2.0, Python >=3.10 | [Release](https://pypi.org/project/setuptools/84.2.0/), MIT | Build backend в pyproject.toml; build не запускался |
| Safetensors | 0.6.2 release metadata: Python >=3.9; PyTorch extra declares torch>=1.10 | [PyPI 0.6.2 metadata](https://pypi.org/pypi/safetensors/0.6.2/json), [v0.6.2 Apache-2.0 license](https://github.com/huggingface/safetensors/blob/v0.6.2/LICENSE), [official Torch API](https://huggingface.co/docs/safetensors/api/torch) | Pinned direct runtime dependency; compatible with project Python floor and documented torch lower bound. GM version remains UNKNOWN; package not installed |
| CUDA/NCCL/drivers | Версии GM UNKNOWN | Управляются GM отдельно | Не добавлять системные компоненты в requirements |

## VERIFIED — API evidence, не runtime admission

- [torch.argsort 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.argsort.html): stable=True сохраняет порядок равных значений. При начальной оси expert IDs по возрастанию это кандидат для tie-breaking D03. Stable ranking не гарантирует bitwise-identical GEMM на разных devices.
- [torch.utils.checkpoint 2.14](https://docs.pytorch.org/docs/2.14/checkpoint.html): non-reentrant вариант принимает context_fn для разных original/recompute contexts. Side effects routing нельзя полагаться повторять один раз автоматически.
- [torch.distributed 2.14](https://docs.pytorch.org/docs/2.14/distributed.html): all_reduce и all_to_all_single документированы. all_to_all_single помечен experimental; explicit split sizes нужны для variable assignments. Существование API не доказывает autograd semantics и поддержку backend.
- [torch.optim 2.14](https://docs.pytorch.org/docs/2.14/optim.html): optimizer interfaces документированы; optimizer и его coefficients для SHINRA не выбраны.

## PROPOSED import graph

Текущие imports существующих модулей проверены статически: внешние roots — `yaml`, `torch` и лениво импортируемый `safetensors`; внутренние imports идут через `shinra`, остальное — stdlib. Safetensors checkpoint adapter теперь вызывает официальные `safetensors.torch.save_file` и `load_file`; проектная зависимость закреплена на 0.6.2. Этот релиз объявляет Python >=3.9 и его `torch` extra — torch>=1.10. Поскольку managed GM PyTorch version неизвестна, реальная совместимость и deployment admission остаются unverified; ничего не устанавливалось.

PyYAML SafeLoader сам по себе не является всей validation: SHINRA обязана отдельно запрещать duplicate keys, aliases/merge keys, multiple documents и неизвестные fields, а также исключать YAML implicit typing из скрытого изменения config semantics. Никаких unsafe constructors.

## Excluded / not admitted

FlashAttention, Triton, DeepGEMM, Transformer Engine, TensorRT не приняты как backend, не добавляются imports/dependencies. Hugging Face tokenizer/model packages не нужны для собственного BPE/weights. Стандартные корпуса и download clients не подключаются. Test packages запрещены.

## UNKNOWN / BLOCKED

Точные GM Python/PyTorch/CUDA versions, ABI и collective support UNKNOWN. Исходник attention содержит native torch_blockwise_exact, выбор backend в YAML оставлен явным null. Safetensors serialization/manifest/recovery source exists; storage durability and actual GM compatibility are not verified. Документальный API target 2.14 не означает требование переустановить GM. Dependency was pinned but not installed; execution remains prohibited.

## Static source evidence — 25.10.2026

AST-разбор 25 Python-файлов без импорта модулей. Прямые внешние import roots: yaml, torch и safetensors. Остальные roots — shinra и стандартная библиотека. Attention backend реализует blockwise online softmax и собственный backward; не является принятым GPU kernel backend и не имеет измерений скорости или runtime correctness. Higher-order gradients не поддерживаются. Checkpoint serialization и recovery реализованы на source level; backend durability, реальная интеграция model/optimizer state и runtime correctness не проверены.

## Dtype admission correction — 25.10.2026

The pinned [v0.6.2 Torch adapter](https://github.com/huggingface/safetensors/blob/v0.6.2/bindings/python/py_src/safetensors/torch.py) lists float8_e4m3fn and float8_e5m2 but not FNUZ variants. SHINRA now rejects F8_E4M3FNUZ/F8_E5M2FNUZ at inventory/manifest admission, without casting. This does not enable FP8 model execution. Requirements unchanged; no package installed. Actual GM compatibility remains UNKNOWN.

## Order 007 API review — 26.10.2026

No new external dependency added. New algorithms use stdlib and the externally managed Torch stack. Official documentation reviewed: [AdamW 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html), [distributed 2.14](https://docs.pytorch.org/docs/2.14/distributed.html), [non-reentrant checkpoint 2.14](https://docs.pytorch.org/docs/2.14/checkpoint.html). Calls include all_gather, all_reduce, all_to_all_single, broadcast, process-group membership, AdamW and checkpoint(context_fn, use_reentrant=False). Explicit custom autograd pairs communication; documentation does not establish runtime correctness. Installed GM version/backend support remains UNKNOWN. No installation or version import performed. Existing license register and pinned requirements remain unchanged.

## Order 008 API review — 26.10.2026

Rechecked [PyTorch checkpoint 2.14](https://docs.pytorch.org/docs/2.14/checkpoint.html) and [distributed 2.14](https://docs.pytorch.org/docs/2.14/distributed.html). set_checkpoint_early_stop(False) surrounds original checkpoint forward; explicit collectives remain caller-group owned. Actual GM compatibility remains UNKNOWN; no dependency added or installed.


## Order 009 transport API review — 27.10.2026

Official [PyTorch distributed 2.14 documentation](https://docs.pytorch.org/docs/2.14/distributed.html) reviewed for all_gather/all_reduce/broadcast, isend/irecv and Work.wait. P2P src/dst use global ranks with explicit process groups; NCCL tags are not used. All sends are waited before crossing to another group. Caller must create groups in consistent order and select the actual current CUDA device before NCCL operations; Group validates tensor/current-device agreement. Gloo adapter admits CPU only. Nonempty payload edges avoid assuming zero-count P2P support. Native custom autograd reverses EP transport and implements CP/TP adjoints; this is source semantics, not backend certification. Actual installed GM version, backend behavior and timeout/restart behavior remain UNKNOWN. No new dependency or installation.
