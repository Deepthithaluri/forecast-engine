import plotly.express as px
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="AI Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
)

st.title("📊 AI Sales Forecasting Dashboard")

st.markdown(
    """
Welcome to the **AI-Powered Sales Forecasting &
Product Recommendation System**.
"""
)

st.divider()

col1, col2, col3 = st.columns(3)

col1.metric(
    "Forecast API",
    "🟢 Running",
)

col2.metric(
    "Recommendation API",
    "🟢 Running",
)

col3.metric(
    "Backend",
    "Healthy",
)

st.divider()

st.subheader("Project Overview")

st.write(
    """
This project predicts future sales using Machine Learning
and recommends products using a co-occurrence-based
recommendation engine.
"""
)

st.subheader("Technology Stack")

st.markdown("""
- 🚀 FastAPI
- 📈 Streamlit
- 🤖 XGBoost
- 🐼 Pandas
- 📊 Plotly
- 💾 Joblib
""")

st.divider()

st.subheader("Sample Sales Trend")

df = pd.DataFrame({
    "Day":[1,2,3,4,5,6,7],
    "Sales":[1200,1400,1350,1600,1700,1800,1900]
})

fig = px.line(
    df,
    x="Day",
    y="Sales",
    markers=True,
)

st.plotly_chart(
    fig,
    use_container_width=True,
)