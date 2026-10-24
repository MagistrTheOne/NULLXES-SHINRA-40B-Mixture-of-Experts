# D09 — Dependency register

Проверка первичных источников: 24.10.2026. Исходников ещё нет, поэтому фактический import graph отсутствует. Ни один пакет не установлен и requirements.txt не создан. Ниже проектные потребности, а не утверждение совместимости GM.

| Component | Version evidence | License/source | Admission |
|---|---|---|---|
| PyTorch, managed by GM | Документация 2.14 просмотрена; фактическая версия GM и минимальная поддерживаемая версия SHINRA UNKNOWN | [Official LICENSE](https://github.com/pytorch/pytorch/blob/main/LICENSE), BSD-style terms и third-party notices | Базовая потребность V3.0; compatibility UNKNOWN; не добавлять torch в requirements |
| PyYAML | 6.0.3 опубликован, Python >=3.8 по metadata | [Official project release](https://pypi.org/project/PyYAML/6.0.3/), MIT | PROPOSED прямой parser dependency, не установлен/не принят автоматически |
| Python stdlib | Версия GM UNKNOWN | Поставляется с интерпретатором | json/hashlib/pathlib/dataclasses/enum/typing/codecs и др. не помещаются в requirements |
| Packaging backend | UNKNOWN | Не выбран | BLOCKED выбор build-system до реализации; не выдумывать pin |
| CUDA/NCCL/drivers | Версии GM UNKNOWN | Управляются GM отдельно | Не добавлять системные компоненты в requirements |

## VERIFIED — API evidence, не runtime admission

- [torch.argsort 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.argsort.html): stable=True сохраняет порядок равных значений. При начальной оси expert IDs по возрастанию это кандидат для tie-breaking D03. Stable ranking не гарантирует bitwise-identical GEMM на разных devices.
- [torch.utils.checkpoint 2.14](https://docs.pytorch.org/docs/2.14/checkpoint.html): non-reentrant вариант принимает context_fn для разных original/recompute contexts. Side effects routing нельзя полагаться повторять один раз автоматически.
- [torch.distributed 2.14](https://docs.pytorch.org/docs/2.14/distributed.html): all_reduce и all_to_all_single документированы. all_to_all_single помечен experimental; explicit split sizes нужны для variable assignments. Существование API не доказывает autograd semantics и поддержку backend.
- [torch.optim 2.14](https://docs.pytorch.org/docs/2.14/optim.html): optimizer interfaces документированы; optimizer и его coefficients для SHINRA не выбраны.

## PROPOSED import graph

core parser → yaml + stdlib; tokenizer/data metadata → stdlib; model/moe/training tensor operations → torch; distributed → torch.distributed; checkpoint payload adapter UNKNOWN. Actual imports нужно сверить по каждому исходному файлу, когда он появится. До этого нельзя утверждать окончательный requirements.

PyYAML SafeLoader сам по себе не является всей validation: SHINRA обязана отдельно запрещать duplicate keys, aliases/merge keys, multiple documents и неизвестные fields, а также исключать YAML implicit typing из скрытого изменения config semantics. Никаких unsafe constructors.

## Excluded / not admitted

FlashAttention, Triton, DeepGEMM, Transformer Engine, TensorRT не приняты как backend, не добавляются imports/dependencies. Hugging Face tokenizer/model packages не нужны для собственного BPE/weights. Стандартные корпуса и download clients не подключаются. Test packages запрещены.

## UNKNOWN / BLOCKED

Точные GM Python/PyTorch/CUDA versions, ABI, выбранный attention backend, collective support и checkpoint payload format UNKNOWN. Документальный API target 2.14 не означает требование переустановить GM. Реализация зависимых adapters BLOCKED до конкретного выбора; static metadata и native byte algorithms независимы от runtime GPU stack.
