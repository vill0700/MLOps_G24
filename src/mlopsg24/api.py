import gc
from contextlib import asynccontextmanager
from dataclasses import asdict
from http import HTTPStatus
from datetime import datetime
import re

import torch
from fastapi import BackgroundTasks, FastAPI
from loguru import logger

from mlopsg24.inference import InferenceClassify, DataPrediction


@asynccontextmanager
async def levetid(app: FastAPI):
    """
    - Loads a pretrained hugginfae model into FastAPI once at first call.
    - At shutdown of API, the hf model is deleted and memory is released.
    """
    app.state.inferencer = InferenceClassify()
    logger.info("instance of InferenceClassify() loaded into FastAPI app.state")

    yield

    del app.state.inferencer
    gc.collect()
    logger.info("succesfully closed. Deleted instance of InferenceClassify(). Cleared GPU - just in case")


app = FastAPI(lifespan=levetid)

def add_to_database(dataclass_prediction:DataPrediction, jobopslag:str):
    """Add record to databas of predictions"""
    # Simple example of a database record
    # For proper setup it should be record to a DB table and
    # contain input or id to ID to input jobopslag
    now = str(datetime.now())
    clean_str = re.sub(r'[^a-zA-Z ]', '', jobopslag)
    with open("data/drift/prediction_records.csv", "a") as file:
        file.write(f"{now}, {clean_str[:100]}, {dataclass_prediction.categori_label}, {dataclass_prediction.categori_idx}\n")


@app.get("/")
def health_check():
    """Health check."""
    response = {
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response


@app.post("/classify")
def predict(jobopslag: str, background_task:BackgroundTasks) -> dict:
    """
    Makes a prediction of type DataPrediction.
    Insert a record into a database to enable monitoring of data drift.
    Return a json/dict to the user
    """

    dataclass_prediction = app.state.inferencer.classify(jobopslag)

    background_task.add_task(add_to_database, dataclass_prediction, jobopslag)

    return asdict(dataclass_prediction)
