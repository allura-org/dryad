from litestar import Router, get

def create_router(host: str, port: int) -> Router:
    @get("/.well-known/serviceinfo")
    async def serviceinfo() -> dict:
        return {
            "version": 0.1,
            "software": {
                "name": "Dryad",
                "version": "0.1.0",
                "repository": "https://github.com/allura-org/dryad",
                "homepage": "https://github.com/allura-org/dryad",
            },
            "api": {
                "openai": {
                    "name": "OpenAI API",
                    "base_url": f"http://{host}:{port}/v1",
                    "documentation": f"http://{host}:{port}/docs",
                    "version": 1,
                }
            }
        }

    return Router(route_handlers=[serviceinfo])
