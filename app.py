import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Page Config
st.set_page_config(
    page_title="Intelligent Solar Forecasting",
    page_icon="🌞",
    layout="wide"
)

# Header
st.markdown("""
# Intelligent Solar Energy Forecasting
### From Data Analytics to Renewable Energy Insights
""")

st.markdown("---")

# Sidebar
st.sidebar.header("📂 Upload Data Files")

gen_file = st.sidebar.file_uploader("Upload Generation Data", type=["csv"])
weather_file = st.sidebar.file_uploader("Upload Weather Data", type=["csv"])


#would work even if no fiels are uploaded
if gen_file is None or weather_file is None:
    st.sidebar.info("Using sample dataset")
    gen = pd.read_csv("data/sample_generation.csv")
    weather = pd.read_csv("data/sample_weather.csv")
else:
    gen = pd.read_csv(gen_file)
    weather = pd.read_csv(weather_file)


st.sidebar.markdown("---")
st.sidebar.info("Model: Random Forest Regressor\n\nFeatures: Weather + Temporal + Lag")

if gen_file and weather_file:

with st.spinner("Processing and Training Model..."):

    # Convert date
    gen['DATE_TIME'] = pd.to_datetime(gen['DATE_TIME'])
    weather['DATE_TIME'] = pd.to_datetime(weather['DATE_TIME'])

    df = pd.merge(gen, weather, on='DATE_TIME')
    df = df.sort_values('DATE_TIME')
    df = df.fillna(method='ffill')

    # Feature Engineering
    df['hour'] = df['DATE_TIME'].dt.hour
    df['month'] = df['DATE_TIME'].dt.month

    df['prev_1'] = df['AC_POWER'].shift(1)
    df['prev_2'] = df['AC_POWER'].shift(2)
    df['prev_3'] = df['AC_POWER'].shift(3)

    df = df.dropna()

    features = [
        'AMBIENT_TEMPERATURE',
        'MODULE_TEMPERATURE',
        'IRRADIATION',
        'hour',
        'month',
        'prev_1',
        'prev_2',
        'prev_3'
    ]

    X = df[features]
    y = df['AC_POWER']

    # SCALING (not necessary)
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)


    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    st.success("Model Training Complete ")

    # Metrics Section
    st.markdown("##  Model Performance")
    col1, col2 = st.columns(2)

    col1.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")
    col2.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f}")

    st.markdown("---")

    # Forecast Graph
    st.markdown("## Forecast vs Actual Power Output")

    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(y_test.values[:300], label="Actual", linewidth=2)
    ax.plot(predictions[:300], label="Predicted", linestyle="--")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("AC Power")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("---")

    # Feature Importance
    st.markdown("##  Feature Importance")

    fig2, ax2 = plt.subplots(figsize=(8,5))
    ax2.barh(features, model.feature_importances_)
    ax2.set_xlabel("Importance Score")
    st.pyplot(fig2)

    st.markdown("---")

    # Seasonality
    st.markdown("##  Average Power by Hour")

    hourly_avg = df.groupby('hour')['AC_POWER'].mean()

    fig3, ax3 = plt.subplots(figsize=(8,5))
    hourly_avg.plot(ax=ax3)
    ax3.set_ylabel("Average AC Power")
    ax3.grid(True)
    st.pyplot(fig3)

    st.markdown("---")

    st.markdown("""
    ### 🔬 Key Insights
    - Solar generation strongly depends on irradiation levels.
    - Peak generation observed during midday hours.
    - Lag features improved predictive stability.
    - Model captures non-linear weather relationships effectively.
    """)
