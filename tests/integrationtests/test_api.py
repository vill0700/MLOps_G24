from fastapi.testclient import TestClient

from mlopsg24.api import app

# NOTE: Contex manager with TestClient() is used to ensure that the app runs it lifetime section.

def test_health_check():
    """
    This test is meant to test if the app will respond with the proper root message
    """
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"message": "OK", "status-code": 200}


def test_predict():
    """
    This test is meant to test that the api returns outputs in the expectied format
    """

    with TestClient(app) as client:
        mock_jobopslag = (
            "Du er pædagog og vant til at arbejde med børn, der har brug for "
            "tydelige rammer og forudsigelighed i hverdagen. Du formår at "
            "skabe ro og nærvær i relationen og møder barnet med forståelse."
        )

        response = client.post("/classify", params={"jobopslag": mock_jobopslag})

        assert isinstance(response.json()["categori_label"], str)
        assert len(response.json()["probability_distribution"]) == 22 , \
            "tests that there are 22 classes in the probability distribution"
        assert sum(response.json()["probability_distribution"]) > 0.9999, \
            "tests that the probability distributions sums to almost 100% -\
            why? hint: statistics and floating point numbers"
