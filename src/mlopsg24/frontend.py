import streamlit as st
from fastapi.testclient import TestClient

# import sys
# sys.path.append("/data/projects/overvaag/ESHA/mlops_course/MLOps_G24/")

from src.mlopsg24.api import api



@st.cache_resource
def get_localhost_client():
    """Create and cache the TestClient instance"""
    return TestClient(api)


def call_classification_api(jobopslag: str):
    """
    Call the FastAPI /classify endpoint using TestClient

    Args:
        jobopslag: Job posting text to classify

    Returns:
        dict: API response or None if error
    """
    with TestClient(api) as client:
        client = get_localhost_client()
        response = client.get("/classify", params={"jobopslag": jobopslag})

    return response.json()

# Streamlit UI Configuration
st.set_page_config(
    page_title="Job Klassifikation",
    page_icon="💼",
    layout="wide"
)

st.title("💼 Job Klassifikation Prototype")
st.markdown("Indtast et jobopslag for at klassificere det")

# Create two columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    # Text input area
    jobopslag_input = st.text_area(
        "Jobopslag tekst:",
        value=(
            "Du er pædagog og vant til at arbejde med børn, der har brug for "
            "tydelige rammer og forudsigelighed i hverdagen. Du formår at "
            "skabe ro og nærvær i relationen og møder barnet med forståelse."
        ),
        height=200,
        help="Indtast eller indsæt jobopslag teksten her"
    )

    # Character count
    st.caption(f"Antal tegn: {len(jobopslag_input)}")

    # Action buttons
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])

    with col_btn1:
        classify_button = st.button("🔍 Klassificer", type="primary", use_container_width=True)

    with col_btn2:
        clear_button = st.button("🗑️ Ryd", use_container_width=True)

with col2:
    st.info(
        """
        **Sådan bruges værktøjet:**

        1. Indtast jobopslag tekst
        2. Klik på 'Klassificer'
        3. Se resultaterne nedenfor
        """
    )

# Handle clear button
if clear_button:
    st.rerun()

# Handle classification
if classify_button:
    if jobopslag_input.strip():
        with st.spinner("Klassificerer jobopslag..."):
            result = call_classification_api(jobopslag_input)

        if result:
            st.success("✅ Klassifikation gennemført!")

            # Display results
            st.subheader("Resultater")

            # You can customize this based on your API response structure
            st.json(result)


    else:
        st.warning("⚠️ Indtast venligst et jobopslag før klassifikation")

# Footer
st.divider()
st.caption("Dette er en prototype. FastAPI backend bruges via TestClient.")