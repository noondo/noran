import requests
import streamlit as st

# Configure the Streamlit page layout and title
st.set_page_config(page_title="Breast Cancer Predictor", layout="centered")

st.title("🩺 Breast Cancer Classification")
st.write(
    "Enter the 5 cell nucleus measurements below to predict the tumor class (Malignant or Benign)."
)

# User input fields with default test values
mean_perimeter = st.number_input("Mean Perimeter", value=122.8, format="%.4f")
mean_concave_points = st.number_input(
    "Mean Concave Points", value=0.1471, format="%.4f"
)
worst_radius = st.number_input("Worst Radius", value=25.38, format="%.4f")
worst_perimeter = st.number_input("Worst Perimeter", value=184.6, format="%.4f")
worst_concave_points = st.number_input(
    "Worst Concave Points", value=0.2654, format="%.4f"
)

if st.button("Predict 🚀"):
    # Define the local FastAPI endpoint
    api_url = "http://127.0.0.1:8000/predict"

    # Prepare the payload matching the Pydantic schema
    payload = {
        "mean_perimeter": mean_perimeter,
        "mean_concave_points": mean_concave_points,
        "worst_radius": worst_radius,
        "worst_perimeter": worst_perimeter,
        "worst_concave_points": worst_concave_points,
    }

    try:
        # Send POST request to the API
        response = requests.post(api_url, json=payload)

        if response.status_code == 200:
            result = response.json()
            label = result["label"].capitalize()
            confidence = result.get("confidence_score")

            # Display the prediction result
            if label.lower() == "malignant":
                st.error(f"**Prediction:** {label}")
            else:
                st.success(f"**Prediction:** {label}")

            if confidence:
                st.info(f"**Confidence Score:** {confidence:.2f}%")
        else:
            st.warning(f"API Error ({response.status_code}): {response.text}")

    except requests.exceptions.ConnectionError:
        st.error(
            "❌ Connection Error: Could not reach the API. Please ensure the FastAPI server is running!"
        )
