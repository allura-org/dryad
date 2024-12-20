import llama_cpp
import llama_cpp.llama_tokenizer
import numpy as np
import ctypes
import traceback
from collections.abc import Callable
from llama_cpp import llama_context_p, llama_model_p, llama_model_params, llama_context_params

from batches import Batch
from samplers import Sampler

class LoadedModel:
    def __init__(self, model: llama_model_p, context: llama_context_p, name: str = None):
        self.model = model
        self.context = context
        self.name = name
        self.batch = None

    @staticmethod
    def load_from_file(path: str, model_params: llama_model_params, context_params: llama_context_params, name: str = None) -> "LoadedModel":
        model = llama_cpp.llama_load_model_from_file(bytes(path, "utf-8"), model_params)
        context = llama_cpp.llama_new_context_with_model(model, context_params)
        return LoadedModel(model, context, name)
    
    def get_model_name(self) -> str:
        if self.name is not None:
            return self.name
        name_buf = ctypes.create_string_buffer(1024)
        llama_cpp.llama_model_meta_val_str(self.model, b"general.name", name_buf, 1024)
        return name_buf.value.decode("utf-8")

    def tokenize(self, prompt: str) -> ctypes.Array[llama_cpp.llama_token]:
        tokens = (llama_cpp.llama_token * (len(prompt) + 1))()
        n_tok = llama_cpp.llama_tokenize(
            self.model,
            bytes(prompt, "utf-8"),
            len(prompt),
            tokens,
            len(tokens),
            True,
            True
        )
        return tokens[:n_tok]

    def untokenize(self, token: llama_cpp.llama_token) -> str:
        buf = ctypes.create_string_buffer(1024)
        llama_cpp.llama_token_to_piece(self.model, token, buf, 1024, 0, True)
        return buf.value.decode("utf-8")
    
    def generate(self, prompt: str, n_tokens: int, sampler_fns: dict[str, list[Callable[[np.ndarray], np.ndarray]]] = {}) -> ctypes.Array[str]:
        try:
            return self._generate(prompt, n_tokens, sampler_fns)
        except Exception as e:
            print(traceback.format_exc())
            print(e)
            print("Ran into error, clearing batch")
            self.batch.common_batch_clear()
            del self.batch
            self.batch = None
            return []

    def _generate(self, prompt: str, n_tokens: int, sampler_fns: dict[str, list[Callable[[np.ndarray], np.ndarray]]] = {}) -> ctypes.Array[str]:
        if self.batch is not None:
            return
        tokens = self.tokenize(prompt)
        if len(tokens) == 0:
            return
        n_ctx = llama_cpp.llama_n_ctx(self.context)
        if len(tokens) > n_ctx:
            tokens = tokens[-n_ctx:]
        self.batch = Batch(n_ctx, n_ctx)
        print(tokens)
        print("".join([self.untokenize(token) for token in tokens]))
        self.batch.set_tokens(tokens)
        self.batch.eval(self.context)

        pred_tokens = []
        n_pred = 0
        logits_idx = len(tokens) - 1

        sampler = Sampler(self, sampler_fns)

        while n_pred < n_tokens:
            logits = llama_cpp.llama_get_logits_ith(self.context, logits_idx)
            token = sampler.sample(logits)
            pred_tokens.append(token)
            
            if llama_cpp.llama_token_is_eog(self.model, token):
                print("EOG")
                break
            if n_pred >= n_tokens:
                break

            # debug
            print(f"token: {token}")
            print(f"token (decoded): {self.untokenize(token)}")
            
            logits_idx = logits_idx + 1
            n_pred += 1

            self.batch.common_batch_clear()
            self.batch.set_tokens(tokens + pred_tokens)
            self.batch.eval(self.context)

        self.batch.common_batch_clear()
        del self.batch
        self.batch = None

        return [self.untokenize(token) for token in pred_tokens]
