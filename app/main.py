import uvicorn

from app.api import router
from app.container import Container

container = Container()
app = container.app()

app.include_router(router)


def main() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
