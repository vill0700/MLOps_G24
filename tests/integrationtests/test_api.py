#NOTE:DEV
# import sys
# sys.path.append("/data/projects/overvaag/ESHA/mlops_course/MLOps_G24/")
from fastapi.testclient import TestClient
from mlopsg24.api import app


def test_health_check():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {'message': 'OK', 'status-code': 200}

def test_predict():
    with TestClient(app) as client:

        mock_jobopslag = (
            "Du er pædagog og vant til at arbejde med børn, der har brug for "
            "tydelige rammer og forudsigelighed i hverdagen. Du formår at "
            "skabe ro og nærvær i relationen og møder barnet med forståelse."
        )

        response = client.get("/classify", params={"jobopslag": mock_jobopslag})

        assert isinstance(response.json()["prediction"], str)
        assert len(response.json()["probability distribution"])==22
