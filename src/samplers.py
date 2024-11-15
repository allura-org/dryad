from collections.abc import Callable
import numpy as np
import llama_cpp
import ctypes

def apply_temperature_pre(temperature: float) -> np.ndarray:
    def apply(logits: np.ndarray) -> np.ndarray:
        return logits / temperature
    return apply

def apply_temperature_post(num_candidates: int, temperature: float, min_temp: float = 0.0, max_temp: float = 0.0, temp_exponent: float = 1.0) -> np.ndarray:
    def apply(logits: np.ndarray) -> np.ndarray:
        if max_temp > min_temp:
            # Calculate entropy of the softmax probabilities
            entropy = -np.sum(logits * np.log(logits + 1e-7))  # Ensure no log(0) by adding a small epsilon
            # Calculate maximum possible entropy
            max_entropy = -np.log(1.0 / num_candidates)
            # Guard against division by zero
            if max_entropy == 0.0:
                max_entropy = 1.0
            # Normalize the entropy
            normalized_entropy = entropy / max_entropy
            # Map the normalized entropy to the desired temperature range using the power function
            temperature = min_temp + (max_temp - min_temp) * (normalized_entropy ** temp_exponent)
        
        # Apply the temperature to the softmax probabilities
        itemp = 1.0 / temperature
        logits = np.power(logits, itemp)
        # Normalize them again
        logits /= np.sum(logits)
        
        return logits
    return apply

class Sampler:
    def __init__(self, model, sampler_fns: dict[str, list[Callable[[np.ndarray], np.ndarray]]]):
        self.model = model
        self.rng = np.random.default_rng()
        self.sampler_fns = sampler_fns

    def sample(self, logits: ctypes.Array[float]) -> llama_cpp.llama_token:
        logits_np = np.ctypeslib.as_array(logits, (llama_cpp.llama_n_vocab(self.model.model),))

        for fn in self.sampler_fns["before_softmax"]:
            logits_np = fn(logits_np)

        softmax_logits = np.exp(logits_np - np.max(logits_np))
        softmax_logits /= softmax_logits.sum()

        for fn in self.sampler_fns["after_softmax"]:
            softmax_logits = fn(softmax_logits)

        idx = self.rng.choice(len(softmax_logits), p=softmax_logits, size=1)[0]
        return idx