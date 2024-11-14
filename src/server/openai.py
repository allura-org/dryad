from litestar import Router, get

from model import LoadedModel

def create_router(model: LoadedModel) -> Router:
    @get("/models")
    async def models() -> dict:
        return {
            "object": "list",
            "data": [
                {"id": model.get_model_name(), "object": "model", "owned_by": "dryad"}
            ]
        }

    return Router(path="/v1", route_handlers=[models])
