import llama_cpp
import ctypes
from llama_cpp import llama_context_p, llama_model_p, llama_model_params, llama_context_params

class LoadedModel:
    def __init__(self, model: llama_model_p, context: llama_context_p, name: str = None):
        self.model = model
        self.context = context
        self.name = name
    
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
