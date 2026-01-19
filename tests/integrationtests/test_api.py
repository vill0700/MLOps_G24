from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mlopsg24.api import app
from mlopsg24.inference import DataPrediction

path_gliner2 = Path("models/fastino/gliner2-multi-v1")
path_e5 = Path("models/intfloat/multilingual-e5-large-instruct")

pretrained_models_available = all((
    Path("models/fastino/gliner2-multi-v1").exists(),
    Path("models/intfloat/multilingual-e5-large-instruct").exists(),
))

@pytest.mark.skipif(
    not pretrained_models_available,
    reason=(
        "pretrained huggingface text models are not available."
        "github does not have access to data file so CI will fail"
    ),
)
def test_health_check():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "OK", "status-code": 200}


@pytest.mark.skipif(
    not pretrained_models_available,
    reason=(
        "pretrained huggingface text models are not available."
        "github does not have access to data file so CI will fail"
    ),
)
def test_predict():
    with TestClient(app) as client:
        mock_jobopslag = (
            "Du er pædagog og vant til at arbejde med børn, der har brug for "
            "tydelige rammer og forudsigelighed i hverdagen. Du formår at "
            "skabe ro og nærvær i relationen og møder barnet med forståelse."
        )

        response = client.get("/classify", params={"jobopslag": mock_jobopslag})

        assert isinstance(response.json()["categori_label"], str)
        assert len(response.json()["probability_distribution"]) == 22 , \
            "tests that there are 22 classes in the probability distribution"
        assert sum(response.json()["probability_distribution"]) > 0.9999, \
            "tests that the probability distributions sums to almost 100% -\
            why? hint: statistics and floating point numbers"
