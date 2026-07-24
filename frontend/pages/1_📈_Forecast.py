import streamlit as st

from api import forecast

st.title("📈 Sales Forecast")

lag_1 = st.number_input("Lag 1", value=1500.0)

lag_7 = st.number_input("Lag 7", value=1450.0)

rolling_mean_7 = st.number_input(
    "Rolling Mean 7",
    value=1480.0,
)

rolling_std_7 = st.number_input(
    "Rolling Std 7",
    value=120.0,
)

month = st.number_input(
    "Month",
    min_value=1,
    max_value=12,
    value=7,
)

day = st.number_input(
    "Day",
    min_value=1,
    max_value=31,
    value=7,
)

weekday = st.number_input(
    "Weekday",
    min_value=0,
    max_value=6,
    value=1,
)

if st.button("Predict Sales"):

    response = forecast(
        {
            "lag_1": lag_1,
            "lag_7": lag_7,
            "rolling_mean_7": rolling_mean_7,
            "rolling_std_7": rolling_std_7,
            "month": month,
            "day": day,
            "weekday": weekday,
        }
    )

    st.success(
        f"Predicted Sales : {response['predicted_sales']:.2f}"
    )