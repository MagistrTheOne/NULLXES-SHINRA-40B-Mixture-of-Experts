"""Explicit greedy/categorical sampling with stable token-ID ties."""
import torch


def sample(logits, config, *, generator: torch.Generator | None):
    if logits.shape != (131072,) or not bool(torch.isfinite(logits).all()):
        raise ValueError("sampling requires finite full-vocabulary logits")
    method = config.get("inference.sampling")
    if method == "greedy":
        return int(torch.argmax(logits).item())
    if method != "categorical" or generator is None:
        raise ValueError("explicit supported sampler and RNG required")
    temperature, top_k, top_p = (config.get(f"inference.{key}") for key in ("temperature", "top_k", "top_p"))
    if any(x is None for x in (temperature, top_k, top_p)):
        raise ValueError("sampling parameters UNKNOWN")
    with torch.autocast(device_type=logits.device.type, enabled=False):
        scores = logits.float() / temperature
        order = torch.argsort(scores, descending=True, stable=True)
        sorted_scores = scores[order]
        if top_k:
            sorted_scores = sorted_scores[:top_k]
            order = order[:top_k]
        probabilities = torch.softmax(sorted_scores, dim=0)
        if top_p < 1:
            discard = probabilities.cumsum(dim=0) - probabilities >= top_p
            probabilities = probabilities.masked_fill(discard, 0)
            probabilities = probabilities / probabilities.sum()
        if not bool(torch.isfinite(probabilities).all()):
            raise FloatingPointError("nonfinite sampling distribution")
        selected = torch.multinomial(probabilities, 1, replacement=False, generator=generator)
    return int(order[selected].item())
