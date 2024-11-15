import llama_cpp
import uvicorn
import ctypes
from model import LoadedModel
from server.openai import create_router as create_openai_router
from litestar import Litestar
from litestar.openapi.config import OpenAPIConfig
from litestar.config.cors import CORSConfig
from gguf.constants import GGMLQuantizationType
from constants import GGML_LOG_LEVEL

def main(
    model: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    context_length: int = 0,
    gpu_layers: int = 0,
    n_threads: int = 1,
    mmap: bool = False,
    serve_openai: bool = True,
    serve_kobold: bool = True,
    kv_cache_type: GGMLQuantizationType = GGMLQuantizationType.F16,
    offload_kqv: bool = False,
    log_level: GGML_LOG_LEVEL = GGML_LOG_LEVEL.GGML_LOG_LEVEL_INFO,
):
    """
    Dryad is a simple, fast, and extensible single-user LLM server based on llama.cpp.

    Args:
        model (str): The path to the GGUF model file.
    
    Flags:
        context_length (int): The context length to use for the model (defaults to 0, which means the model's default context length).
        gpu_layers (int): The number of layers to keep in VRAM (defaults to 0, which means the model will be loaded onto CPU RAM).
        mmap (bool): Whether to use memory-mapped files for the model (defaults to False).
        serve_openai (bool): Whether to serve the OpenAI API (defaults to True).
        serve_kobold (bool): Whether to serve the KoboldAI API (defaults to True).
        kv_cache_type (GGMLQuantizationType): The type of quantization to use for the KV cache (defaults to F16).
        offload_kqv (bool): Whether to offload the KV cache to the GPU (defaults to True).
    """
    llama_cpp.llama_backend_init(None)

    def log_callback(level: int, message: ctypes.c_char_p, last_log_success: ctypes.c_void_p):
        last_log_success_2 = ctypes.cast(last_log_success, ctypes.POINTER(ctypes.c_int))
        if last_log_success_2.contents.value == 0 and level == GGML_LOG_LEVEL.GGML_LOG_LEVEL_CONT.value[0]:
            return 0
        if level >= log_level.value[0]:
            print(f"{message.decode('utf-8').strip()}")
            last_log_success_2.contents.value = 1
        else:
            last_log_success_2.contents.value = 0
        return 0

    LOGFUNC = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int))
    LOGFUNC_PTR = LOGFUNC(log_callback)

    llama_cpp.llama_log_set(LOGFUNC_PTR, ctypes.byref(ctypes.pointer(ctypes.c_int(0))))
    

    # Load the model
    model_params = llama_cpp.llama_model_params()
    model_params.n_gpu_layers = gpu_layers
    model_params.use_mmap = mmap
    model_params.n_threads = n_threads
    context_params = llama_cpp.llama_context_params()
    context_params.n_ctx = context_length
    context_params.n_batch = 1
    context_params.type_k = kv_cache_type
    context_params.type_v = kv_cache_type
    context_params.offload_kqv = offload_kqv
    loaded_model = LoadedModel.load_from_file(model, model_params, context_params, name=model)

    llama_cpp.llama_set_n_threads(loaded_model.context, n_threads, n_threads)

    # Create the server
    routers = []
    if serve_openai:
        routers.append(create_openai_router(loaded_model))
    if serve_kobold:
        # TODO: Implement KoboldAI API
        # routers.append(create_kobold_router(loaded_model))
        pass

    cors_config = CORSConfig(allow_origins=["*"])
    app = Litestar(route_handlers=routers, openapi_config=OpenAPIConfig(title="Dryad", version="0.1.0"), cors_config=cors_config, debug=True)

    uvicorn.run(app, host=host, port=port, log_level="debug")