import argparse
import atexit
from pathlib import Path

import polars as pl
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


def call_classification_api(jobopslag: str, localhost:bool=False) -> dict:
    if localhost:
        client = get_localhost_api_client()
    else:
        client = get_localhost_api_client() #TODO: skift denne ud med gcloud client når den er deployed til cloud
    response = client.post("/classify", params={"jobopslag": jobopslag})
    return response.json()


if __name__ == '__main__':

    # CLI arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--localhost",
        action="store_true",
        help="Set to use localhost instead of the default cloud",
    )
    args = parser.parse_args()


    # load inference map from idx to name string
    # NOTE: should be defined central instead in a config
    category_mapping = pl.read_parquet(Path("data/processed/category_mapping.parquet"))

    # Streamlit UI Configuration
    st.set_page_config(page_title="Job Klassifikation", layout="wide")

    st.title("Job Klassifikation Prototype")
    st.markdown("Indtast et jobopslag for at klassificere det til jobtype")

    jobopslag_input = st.text_area(
        "Jobopslag tekst:",
        value=(
            "Dette er et eksempel: "
            "Du er pædagog og vant til at arbejde med børn, der har brug for "
            "tydelige rammer og forudsigelighed i hverdagen. Du formår at "
            "skabe ro og nærvær i relationen og møder barnet med forståelse."
        ),
        height=200,
        help="Indtast jobopslag tekst her",
    )

    if st.button("Klassificer"):  # NOTE: gør at kodes køres når knap klikkes
        with st.spinner("Klassificerer jobopslag..."):
            result = call_classification_api(jobopslag=jobopslag_input, localhost=args.localhost)
            if result["frontend_error_message"]:
                st.error(result["frontend_error_message"])
            else:
                st.subheader("Predicted Job Type")
                st.success(result['categori_label'])

                st.subheader("Probability distribution")

                df_probability_distribution = pl.DataFrame(
                    data={
                        "categories":category_mapping['categori'],
                        "probability":result['probability_distribution']
                    }
                )

                st.bar_chart(
                    data=df_probability_distribution,
                    x="categories",
                    y="probability",
                    height=800,
                    width=800,
                    horizontal=True,
                )
