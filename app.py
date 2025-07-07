import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Insurance Risk Dashboard", layout="wide")
st.title("📊 Insurance Portfolio Risk Management Dashboard")

# --- Load Model ---
@st.cache_resource
def load_model():
    return joblib.load("risk_model (2).pkl")

model = load_model()

# --- Sidebar File Upload ---
st.sidebar.header("Upload Policy Data")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# --- Main Logic ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    try:
        # --- Feature Engineering ---
        df["Start_Date"] = pd.to_datetime(df["Start_Date"], dayfirst=True, errors="coerce")
        df["Policy_Age_Days"] = (pd.to_datetime("today") - df["Start_Date"]).dt.days

        # --- One-hot encode categorical variables ---
        df_encoded = pd.get_dummies(df, columns=["Policy_Type", "Region"], drop_first=False)

        # --- Align columns to match model's expected input ---
        expected_cols = model.feature_names_in_  # sklearn >= 1.0
        for col in expected_cols:
            if col not in df_encoded.columns:
                df_encoded[col] = 0  # Add missing columns
        X = df_encoded[expected_cols]  # Reorder correctly

        # --- Prediction ---
        preds = model.predict(X)
        # Normalize predictions to 0–1 range safely
        min_score = preds.min()
        max_score = preds.max()
        normalized_scores = (preds - min_score) / (max_score - min_score)
        df["Predicted_Risk_Score"] = np.round(normalized_scores, 3)


        # --- Row Slider ---
        st.subheader("📄 Uploaded Data with Predicted Risk")
        num_rows = st.slider("Select number of rows to preview", min_value=10, max_value=len(df), value=100, step=10)
        st.dataframe(df.head(num_rows))

        # --- Download Button ---
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Data with Risk Scores",
            data=csv,
            file_name="predicted_policies.csv",
            mime="text/csv"
        )

        # --- Risk Distribution ---
        st.subheader("📈 Risk Score Distribution")
        fig = px.histogram(df, x="Predicted_Risk_Score", nbins=30, color_discrete_sequence=["teal"])
        st.plotly_chart(fig, use_container_width=True)

        # --- Aggregated Risk ---
        st.subheader("📍 Average Risk by Region and Policy Type")
        agg = df.groupby(["Region", "Policy_Type"])["Predicted_Risk_Score"].mean().reset_index()
        fig2 = px.bar(agg, x="Region", y="Predicted_Risk_Score", color="Policy_Type", barmode="group")
        st.plotly_chart(fig2, use_container_width=True)

        # --- High-Risk Policies ---
        st.subheader("🚨 High-Risk Policies (Risk > 0.7)")
        high_risk_df = df[df["Predicted_Risk_Score"] > 0.7]
        st.write(f"Total high-risk policies: {len(high_risk_df)}")
        st.dataframe(high_risk_df)

    except Exception as e:
        st.error(f"❌ Error processing file: {e}")

else:
    st.info("👈 Upload a CSV file from sidebar to begin.")

# --- Footer ---
st.markdown("---")
st.caption("Built for Insurance Risk Monitoring | ML + Streamlit Dashboard")
