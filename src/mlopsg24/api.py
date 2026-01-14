from fastapi import FastAPI
from contextlib import asynccontextmanager
from loguru import logger

@asynccontextmanager
async def levetid(api: FastAPI):
    logger.error("Halløjsa")
    print("Halløjsa")
    yield
    logger.error("farveller")

api = FastAPI(lifespan=levetid)

@api.get("/itemsc/{item_id}")
def read_item_c(item_id: int):
    result = {"item_id": item_id}
    return result
