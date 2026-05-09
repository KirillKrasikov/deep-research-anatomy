from app.api import router
from app.container import Container

container = Container()
app = container.app()

app.include_router(router)
