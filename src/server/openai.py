from litestar import Router, get, post, Request

from model import LoadedModel
from samplers import apply_temperature_pre

def create_router(model: LoadedModel) -> Router:
    @get("/models")
    async def models() -> dict:
        return {
            "object": "list",
            "data": [
                {"id": model.get_model_name(), "object": "model", "owned_by": "dryad"}
            ]
        }
    
    @post("/completions")
    async def completions(request: Request) -> dict:
        data = await request.json()
        prompt = data["prompt"]
        n_tokens = data["max_tokens"]
        #temperature = data.get("temperature", 1.0)
        temperature = 0.7
        top_k = data.get("top_k", 0)
        top_p = data.get("top_p", 1)

        sampler_fns = {
            "before_softmax": [apply_temperature_pre(temperature)],
            "after_softmax": []
        }

        res = model.generate(prompt, n_tokens, sampler_fns)

        return {
            "id": "cmpl-123",
            "object": "text_completion",
            "created": 1714864800,
            "model": model.get_model_name(),
            "choices": [{"text": "".join(res)}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}
        }

    return Router(path="/v1", route_handlers=[models, completions])
