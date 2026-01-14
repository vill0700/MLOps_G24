from fastapi import FastAPI
from contextlib import asynccontextmanager
from loguru import logger
from http import HTTPStatus
import torch
import gc

from .inference import InferenceClassify

@asynccontextmanager
async def levetid(api: FastAPI):

    global inferencer # feels unpythonic to do global variables
    inferencer = InferenceClassify()
    logger.info("instance of InferenceClassify() loaded")

    yield

    del inferencer
    gc.collect()
    logger.info("succesfully closed. Deleted instance and cleared GPU just in case")


api = FastAPI(lifespan=levetid)


@api.get("/")
def root():
    """ Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@api.get("/classify")
def predict(jobopslag: str):

    mock_distribution, translate = inferencer.classify(jobopslag)
    mock_prediction = translate.get(0,"fail")

    prediction = mock_prediction

    distribution = {translate.get(idx,"mapfail"):float(prop) for idx,prop in enumerate(mock_distribution[:22])}

    response = {
        "prediction": prediction,
        "probability distribution": distribution, #"TODO", #Maybe a new dict of key:values?
        "received text formatted": jobopslag,
    }

    return response
