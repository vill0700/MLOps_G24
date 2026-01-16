import atexit

import streamlit as st
from fastapi.testclient import TestClient

from mlopsg24.api import app


@st.cache_resource
def get_localhost_api_client():
    """Create and cache the TestClient instance"""

    client = TestClient(app)

    client.__enter__()  # Triggers lifespan startup (loads model)

    def cleanup():
        try:
            client.__exit__(None, None, None)  # Triggers lifespan shutdown
        except Exception as e:
            st.warning(f"Error during client cleanup: {e}")

    atexit.register(cleanup)

    return client


def call_classification_api(jobopslag: str):
    client = get_localhost_api_client()  # TODO: skift denne ud med gcloud client når den er deployed til cloud
    response = client.get("/classify", params={"jobopslag": jobopslag})

    return response.json()


if __name__ == "__main__":
    # Streamlit UI Configuration
    st.set_page_config(page_title="Job Klassifikation", layout="wide")

    st.title("Job Klassifikation Prototype")
    st.markdown("Indtast et jobopslag for at klassificere det til jobtype")

    jobopslag_input = st.text_area(
        "Jobopslag tekst:",
        value=(
            "Dette er et eksempel:"
            "Du er pædagog og vant til at arbejde med børn, der har brug for "
            "tydelige rammer og forudsigelighed i hverdagen. Du formår at "
            "skabe ro og nærvær i relationen og møder barnet med forståelse."
        ),
        height=200,
        help="Indtast jobopslag tekst her",
    )

    if st.button("Klassificer"):  # NOTE: gør at kodes køres når knap klikkes
        with st.spinner("Klassificerer jobopslag..."):
            result = call_classification_api(jobopslag_input)
            if result.get("frontend_error_message"):
                st.error(result["frontend_error_message"])
            else:
                st.json(result)
