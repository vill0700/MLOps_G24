import gc
from contextlib import asynccontextmanager
from http import HTTPStatus
import torch
import gc
from dataclasses import asdict
from fastapi import FastAPI
from loguru import logger

from mlopsg24.inference import InferenceClassify

"""API for job vacancy text classification using FastAPI."""

@asynccontextmanager
async def levetid(app: FastAPI):
    global inferencer  # feels unpythonic to do global variables
    inferencer = InferenceClassify()
    logger.info("instance of InferenceClassify() loaded")

    yield

    del inferencer
    gc.collect()
    logger.info("succesfully closed. Deleted instance of InferenceClassify(). Cleared GPU - just in case")


app = FastAPI(lifespan=levetid)


@app.get("/")
def health_check():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.get("/classify")
def predict(jobopslag: str) -> dict:
    dataclass_prediction = inferencer.classify(jobopslag)
    return asdict(dataclass_prediction)
