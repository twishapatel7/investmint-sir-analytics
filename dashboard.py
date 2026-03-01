import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet

# -------------------
# Page config
# -------------------

st.set_page_config(
    page_title="SIR Royalty Income Analytics",
    layout="wide"
)

st.title("SIR Royalty Income Fund Analytics Dashboard")

# -------------------
# Load Data
# -------------------

restaurants = pd.read_csv("restaurants.csv")
daily = pd.read_csv("daily_metrics.csv")

daily["date"] = pd.to_datetime(daily["date"])

df = daily.merge(restaurants, on="restaurant_id")

# aggregate
revenue_by_date = df.groupby("date")["total_sales"].sum().reset_index()
royalty_by_date = df.groupby("date")["royalty_income"].sum().reset_index()

# moving average
revenue_by_date["30_day_avg"] = revenue_by_date["total_sales"].rolling(30).mean()

# -------------------
# Sidebar filters
# -------------------

st.sidebar.header("Filters")

selected_restaurant = st.sidebar.selectbox(
    "Select Restaurant",
    ["All"] + list(df["restaurant_name"].unique())
)

if selected_restaurant != "All":
    df_filtered = df[df["restaurant_name"] == selected_restaurant]
else:
    df_filtered = df.copy()

# -------------------
# Tabs
# -------------------

tab1, tab2 = st.tabs(["Analytics", "Forecasting"])

# =====================================================
# TAB 1 — ANALYTICS
# =====================================================

with tab1:

    st.header("Executive Overview")

    total_revenue = df_filtered["total_sales"].sum()
    total_royalty = df_filtered["royalty_income"].sum()
    avg_order = df_filtered["avg_order_value"].mean()
    total_customers = df_filtered["customers"].sum()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Revenue", f"${total_revenue:,.0f}")
    col2.metric("Total Royalty Income", f"${total_royalty:,.0f}")
    col3.metric("Avg Order Value", f"${avg_order:,.2f}")
    col4.metric("Total Customers", f"{total_customers:,}")

    st.divider()

    # Revenue vs Moving Average
    st.subheader("Revenue Trend with Moving Average")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=revenue_by_date["date"],
        y=revenue_by_date["total_sales"],
        name="Daily Revenue",
        opacity=0.4
    ))

    fig.add_trace(go.Scatter(
        x=revenue_by_date["date"],
        y=revenue_by_date["30_day_avg"],
        name="30-Day Average",
        line=dict(width=3)
    ))

    st.plotly_chart(fig, use_container_width=True)

    # Revenue vs Royalty
    st.subheader("Revenue vs Royalty Income")

    combined = revenue_by_date.merge(
        royalty_by_date,
        on="date"
    )

    fig2 = go.Figure()

    fig2.add_trace(go.Scatter(
        x=combined["date"],
        y=combined["total_sales"],
        name="Revenue"
    ))

    fig2.add_trace(go.Scatter(
        x=combined["date"],
        y=combined["royalty_income"],
        name="Royalty Income"
    ))

    st.plotly_chart(fig2, use_container_width=True)

    colA, colB = st.columns(2)

    # Revenue by Restaurant
    with colA:

        st.subheader("Revenue by Restaurant")

        rest_rev = df_filtered.groupby(
            "restaurant_name"
        )["total_sales"].sum().reset_index()

        fig3 = px.bar(
            rest_rev,
            x="restaurant_name",
            y="total_sales",
            color="total_sales",
            title="Restaurant Performance"
        )

        st.plotly_chart(fig3, use_container_width=True)

    # Pie chart by Brand
    with colB:

        st.subheader("Revenue Distribution by Brand")

        brand_rev = df_filtered.groupby(
            "brand"
        )["total_sales"].sum().reset_index()

        fig4 = px.pie(
            brand_rev,
            names="brand",
            values="total_sales",
            hole=0.4
        )

        st.plotly_chart(fig4, use_container_width=True)

    # Customer trends
    st.subheader("Customer Traffic Over Time")

    cust = df_filtered.groupby("date")["customers"].sum().reset_index()

    fig5 = px.line(
        cust,
        x="date",
        y="customers"
    )

    st.plotly_chart(fig5, use_container_width=True)

# =====================================================
# TAB 2 — FORECASTING
# =====================================================

with tab2:

    st.header("Predictive Forecasting")

    forecast_df = revenue_by_date.rename(
        columns={"date": "ds", "total_sales": "y"}
    )

    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )

    model.fit(forecast_df)

    future = model.make_future_dataframe(periods=90)

    forecast = model.predict(future)

    st.subheader("Revenue Forecast")

    fig6 = go.Figure()

    fig6.add_trace(go.Scatter(
        x=forecast_df["ds"],
        y=forecast_df["y"],
        name="Actual"
    ))

    fig6.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat"],
        name="Forecast"
    ))

    fig6.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_upper"],
        name="Upper Bound",
        line=dict(dash="dash")
    ))

    fig6.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat_lower"],
        name="Lower Bound",
        line=dict(dash="dash")
    ))

    st.plotly_chart(fig6, use_container_width=True)

    forecast["royalty_forecast"] = forecast["yhat"] * 0.06

    st.subheader("Royalty Income Forecast")

    fig7 = px.line(
        forecast,
        x="ds",
        y="royalty_forecast"
    )

    st.plotly_chart(fig7, use_container_width=True)

    next_30 = forecast.tail(30)["royalty_forecast"].sum()

    st.metric(
        "Predicted Royalty Income (Next 30 Days)",
        f"${next_30:,.0f}"
    )