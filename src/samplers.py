from collections.abc import Callable
import numpy as np
import llama_cpp
import ctypes

def apply_temperature_pre(temperature: float) -> np.ndarray:
    def apply(logits: np.ndarray) -> np.ndarray:
        return logits / temperature
    return apply

class Sampler:
    def __init__(self, model, sampler_fns: dict[str, list[Callable[[np.ndarray], np.ndarray]]]):
        self.model = model
        self.rng = np.random.default_rng()
        self.sampler_fns = sampler_fns

    def sample(self, logits: ctypes.Array[float]) -> int:
        logits_np = np.ctypeslib.as_array(logits, (llama_cpp.llama_n_vocab(self.model.model),))

        # Print top 5 tokens and their probabilities
        top_5_indices = np.argpartition(logits_np, -5)[-5:]
        top_5_indices = top_5_indices[np.argsort(logits_np[top_5_indices])][::-1]
        for idx in top_5_indices:
            token_str = self.model.untokenize(idx)
            print(f"Token: {token_str}, Score: {logits_np[idx]:.3f}")

        for fn in self.sampler_fns["before_softmax"]:
            logits_np = fn(logits_np)

        softmax_logits = np.exp(logits_np - np.max(logits_np))
        softmax_logits /= softmax_logits.sum()

        for fn in self.sampler_fns["after_softmax"]:
            softmax_logits = fn(softmax_logits)

        idx = self.rng.choice(len(softmax_logits), p=softmax_logits, size=1)[0]
        return idx