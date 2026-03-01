import streamlit as st
import pandas as pd
import plotly.express as px
from prophet import Prophet

restaurants = pd.read_csv("restaurants.csv")
daily = pd.read_csv("daily_metrics.csv")

daily["date"] = pd.to_datetime(daily["date"])

df = daily.merge(restaurants, on="restaurant_id")

revenue_by_date = df.groupby("date")["total_sales"].sum().reset_index()
royalty_by_date = df.groupby("date")["royalty_income"].sum().reset_index()

st.set_page_config(
    page_title="SIR Royalty Income Dashboard",
    layout="wide"
)

st.title("SIR Royalty Income Fund Analytics")

tab1, tab2 = st.tabs(["Historical Analytics", "Forecasting"])

with tab1:

    st.header("Historical Performance")

    total_revenue = df["total_sales"].sum()
    total_royalty = df["royalty_income"].sum()
    avg_daily_revenue = df["total_sales"].mean()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Revenue", f"${total_revenue:,.0f}")
    col2.metric("Total Royalty Income", f"${total_royalty:,.0f}")
    col3.metric("Avg Daily Revenue", f"${avg_daily_revenue:,.0f}")

    st.subheader("Revenue Over Time")

    fig1 = px.line(
        revenue_by_date,
        x="date",
        y="total_sales",
        title="Daily Revenue"
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.subheader("Royalty Income Over Time")

    fig2 = px.line(
        royalty_by_date,
        x="date",
        y="royalty_income",
        title="Daily Royalty Income"
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Revenue by Restaurant")

    revenue_by_restaurant = df.groupby("restaurant_name")["total_sales"].sum().reset_index()

    fig3 = px.bar(
        revenue_by_restaurant,
        x="restaurant_name",
        y="total_sales",
        title="Total Revenue by Restaurant"
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("30-Day Moving Average")

    revenue_by_date["30_day_avg"] = revenue_by_date["total_sales"].rolling(30).mean()

    fig4 = px.line(
        revenue_by_date,
        x="date",
        y="30_day_avg",
        title="Revenue Trend (Smoothed)"
    )

    st.plotly_chart(fig4, use_container_width=True)


with tab2:

    st.header("Revenue & Royalty Forecast")

    forecast_df = revenue_by_date.rename(columns={
        "date": "ds",
        "total_sales": "y"
    })

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )

    model.fit(forecast_df)

    future = model.make_future_dataframe(periods=90)

    forecast = model.predict(future)

    st.subheader("Revenue Forecast")

    fig5 = px.line(
        forecast,
        x="ds",
        y="yhat",
        title="Predicted Revenue"
    )

    fig5.add_scatter(
        x=forecast_df["ds"],
        y=forecast_df["y"],
        mode="lines",
        name="Actual Revenue"
    )

    st.plotly_chart(fig5, use_container_width=True)

    forecast["royalty_forecast"] = forecast["yhat"] * 0.06

    st.subheader("Royalty Income Forecast")

    fig6 = px.line(
        forecast,
        x="ds",
        y="royalty_forecast",
        title="Predicted Royalty Income"
    )

    st.plotly_chart(fig6, use_container_width=True)

    next_30_days = forecast.tail(30)
    predicted_royalty = next_30_days["royalty_forecast"].sum()

    st.metric(
        "Predicted Royalty Income (Next 30 Days)",
        f"${predicted_royalty:,.0f}"
    )