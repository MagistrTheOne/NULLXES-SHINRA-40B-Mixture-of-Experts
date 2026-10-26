"""BF16 model / FP32 router invariants and caller-owned real RNG snapshots."""
import torch


def validate_model_precision(model):
    for name, parameter in model.named_parameters():
        expected = torch.float32 if name.endswith(".moe.router.weight") else torch.bfloat16
        if parameter.dtype != expected:
            raise ValueError(f"precision contract violated: {name}")


def capture_rng(devices: tuple[int, ...]) -> dict[str, torch.Tensor]:
    if len(set(devices)) != len(devices) or any(type(x) is not int or x < 0 for x in devices):
        raise ValueError("explicit unique CUDA device indices required")
    state = {"torch.cpu": torch.get_rng_state().clone()}
    for device in devices:
        state[f"torch.cuda.{device}"] = torch.cuda.get_rng_state(device).clone()
    return state


def restore_rng(state: dict[str, torch.Tensor], devices: tuple[int, ...]):
    expected = {"torch.cpu"} | {f"torch.cuda.{d}" for d in devices}
    if set(state) != expected:
        raise ValueError("RNG device inventory mismatch")
    for value in state.values():
        if value.dtype != torch.uint8 or value.ndim != 1 or value.device.type != "cpu":
            raise ValueError("expected CPU uint8 RNG state")
    torch.set_rng_state(state["torch.cpu"])
    for device in devices:
        torch.cuda.set_rng_state(state[f"torch.cuda.{device}"], device)
