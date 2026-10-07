import streamlit as st
import asyncio
import json
import pandas as pd
import joblib

from aiocoap import Context
from aiocoap import Message
from aiocoap import GET


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="IoT Air Quality Monitoring",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# LOAD RANDOM FOREST MODEL
# ============================================================

MODEL_FILE = "air_quality_rf_model.pkl"

model = joblib.load(MODEL_FILE)


# ============================================================
# COAP FUNCTION
# ============================================================

async def get_air_quality():

    protocol = await Context.create_client_context()

    request = Message(
        code=GET,
        uri="coap://127.0.0.1:5683/airquality"
    )

    response = await protocol.request(request).response

    data = json.loads(
        response.payload.decode("utf-8")
    )

    return data


# ============================================================
# SYNCHRONOUS WRAPPER
# ============================================================

def get_data_from_coap():

    return asyncio.run(
        get_air_quality()
    )


# ============================================================
# AQI CALCULATION FUNCTIONS
# ============================================================

def calculate_sub_index(value, breakpoints):

    for low, high, ilow, ihigh in breakpoints:

        if low <= value <= high:

            aqi = (
                ((ihigh - ilow) / (high - low))
                * (value - low)
                + ilow
            )

            return round(aqi)

    return None


def calculate_aqi(pm25, pm10, no2, so2, o3):

    # --------------------------------------------------------
    # PM2.5 breakpoints
    # --------------------------------------------------------

    pm25_bp = [
        (0, 30, 0, 50),
        (31, 60, 51, 100),
        (61, 90, 101, 200),
        (91, 120, 201, 300),
        (121, 250, 301, 400),
        (251, 500, 401, 500)
    ]


    # --------------------------------------------------------
    # PM10 breakpoints
    # --------------------------------------------------------

    pm10_bp = [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 250, 101, 200),
        (251, 350, 201, 300),
        (351, 430, 301, 400),
        (431, 500, 401, 500)
    ]


    # --------------------------------------------------------
    # NO2 breakpoints
    # --------------------------------------------------------

    no2_bp = [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 180, 101, 200),
        (181, 280, 201, 300),
        (281, 400, 301, 400),
        (401, 800, 401, 500)
    ]


    # --------------------------------------------------------
    # SO2 breakpoints
    # --------------------------------------------------------

    so2_bp = [
        (0, 40, 0, 50),
        (41, 80, 51, 100),
        (81, 380, 101, 200),
        (381, 800, 201, 300),
        (801, 1600, 301, 400),
        (1601, 2620, 401, 500)
    ]


    # --------------------------------------------------------
    # O3 breakpoints
    # --------------------------------------------------------

    o3_bp = [
        (0, 50, 0, 50),
        (51, 100, 51, 100),
        (101, 168, 101, 200),
        (169, 208, 201, 300),
        (209, 748, 301, 400),
        (749, 1000, 401, 500)
    ]


    # --------------------------------------------------------
    # Calculate individual pollutant indices
    # --------------------------------------------------------

    indices = []

    values_and_breakpoints = [
        (pm25, pm25_bp),
        (pm10, pm10_bp),
        (no2, no2_bp),
        (so2, so2_bp),
        (o3, o3_bp)
    ]


    for value, bp in values_and_breakpoints:

        index = calculate_sub_index(
            value,
            bp
        )

        if index is not None:
            indices.append(index)


    # --------------------------------------------------------
    # AQI = maximum pollutant sub-index
    # --------------------------------------------------------

    if indices:

        return max(indices)

    return None


# ============================================================
# AQI CATEGORY
# ============================================================

def get_aqi_category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Satisfactory"

    elif aqi <= 200:
        return "Moderate"

    elif aqi <= 300:
        return "Poor"

    elif aqi <= 400:
        return "Very Poor"

    else:
        return "Severe"


# ============================================================
# STREAMLIT TITLE
# ============================================================

st.title(
    "🌍 IoT Air Quality Monitoring System"
)

st.write(
    "Live air-quality data is retrieved from the CoAP server "
    "and classified using a Random Forest Machine Learning model."
)


# ============================================================
# GET LIVE DATA BUTTON
# ============================================================

if st.button("🔄 Get Live Air Quality"):

    try:

        # ----------------------------------------------------
        # STEP 1: Get data from CoAP server
        # ----------------------------------------------------

        data = get_data_from_coap()


        # ----------------------------------------------------
        # Check server error
        # ----------------------------------------------------

        if "error" in data:

            st.error(
                "CoAP Server Error: "
                + data["error"]
            )

        else:

            # ------------------------------------------------
            # STEP 2: Display received data
            # ------------------------------------------------

            st.subheader(
                "📡 Live Data Received from CoAP Server"
            )

            st.write(
                "Location:",
                data["location"]
            )

            st.write(
                "Time:",
                data["time"]
            )


            # ------------------------------------------------
            # STEP 3: Extract pollutant values
            # ------------------------------------------------

            pm25 = data["PM2.5"]
            pm10 = data["PM10"]
            no2 = data["NO2"]
            so2 = data["SO2"]
            o3 = data["O3"]


            # ------------------------------------------------
            # STEP 4: Display pollutant values
            # ------------------------------------------------

            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("PM2.5", pm25)

            with col2:
                st.metric("PM10", pm10)

            with col3:
                st.metric("NO₂", no2)

            with col4:
                st.metric("SO₂", so2)

            with col5:
                st.metric("O₃", o3)


            # ------------------------------------------------
            # STEP 5: Prepare ML input
            # ------------------------------------------------

            features = [
                "PM2.5",
                "PM10",
                "NO2",
                "SO2",
                "O3"
            ]

            live_data = pd.DataFrame(
                [[
                    pm25,
                    pm10,
                    no2,
                    so2,
                    o3
                ]],
                columns=features
            )


            # ------------------------------------------------
            # STEP 6: Random Forest Prediction
            # ------------------------------------------------

            prediction = model.predict(
                live_data
            )

            air_quality_status = prediction[0]


            # ------------------------------------------------
            # STEP 7: Calculate AQI
            # ------------------------------------------------

            aqi = calculate_aqi(
                pm25,
                pm10,
                no2,
                so2,
                o3
            )


            aqi_category = get_aqi_category(
                aqi
            )


            # ------------------------------------------------
            # STEP 8: Display AQI
            # ------------------------------------------------

            st.subheader(
                "📊 Air Quality Index"
            )

            aqi_col1, aqi_col2 = st.columns(2)

            with aqi_col1:

                st.metric(
                    "Calculated AQI",
                    aqi
                )

            with aqi_col2:

                st.metric(
                    "AQI Category",
                    aqi_category
                )


            # ------------------------------------------------
            # STEP 9: Random Forest Classification
            # ------------------------------------------------

            st.subheader(
                "🤖 Machine Learning Classification"
            )

            st.success(
                f"Random Forest Prediction: "
                f"{air_quality_status}"
            )


            # ------------------------------------------------
            # STEP 10: Compare AQI and ML result
            # ------------------------------------------------

            st.subheader(
                "🔍 AQI vs Machine Learning"
            )

            comparison = pd.DataFrame({
                "Method": [
                    "AQI Calculation",
                    "Random Forest"
                ],

                "Result": [
                    aqi_category,
                    air_quality_status
                ]
            })

            st.table(
                comparison
            )


            # ------------------------------------------------
            # STEP 11: Display ML Input
            # ------------------------------------------------

            st.subheader(
                "Machine Learning Input"
            )

            st.dataframe(
                live_data,
                use_container_width=True
            )


    except Exception as e:

        st.error(
            f"Error: {str(e)}"
        )