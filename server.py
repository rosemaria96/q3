import asyncio
import json
import requests
import joblib

from aiocoap import resource
from aiocoap import Context
from aiocoap import Message


# =========================================================
# Kochi Coordinates
# =========================================================

LATITUDE = 9.93
LONGITUDE = 76.27


# =========================================================
# Open-Meteo Air Quality API
# =========================================================

API_URL = (
    "https://air-quality-api.open-meteo.com/v1/air-quality"
    f"?latitude={LATITUDE}"
    f"&longitude={LONGITUDE}"
    "&current=pm10,pm2_5,"
    "nitrogen_dioxide,sulphur_dioxide,ozone"
)

print("Open-Meteo API:")
print(API_URL)


# =========================================================
# Load Random Forest Model
# =========================================================

MODEL_FILE = "air_quality_rf_model.pkl"

try:
    model = joblib.load(MODEL_FILE)
    print("\nRandom Forest model loaded successfully.")

except Exception as e:
    model = None
    print("\nWarning: Random Forest model could not be loaded.")
    print("Error:", e)


# =========================================================
# CoAP Air Quality Resource
# =========================================================

class AirQualityResource(resource.Resource):

    async def render_get(self, request):

        print("\n=======================================")
        print("GET request received from CoAP client")
        print("=======================================")

        try:

            # -------------------------------------------------
            # Step 1: Get live data from Open-Meteo
            # -------------------------------------------------

            response = requests.get(API_URL, timeout=10)

            response.raise_for_status()

            data = response.json()

            current = data["current"]


            # -------------------------------------------------
            # Step 2: Extract required parameters
            # -------------------------------------------------

            pm25 = current["pm2_5"]
            pm10 = current["pm10"]
            no2 = current["nitrogen_dioxide"]
            so2 = current["sulphur_dioxide"]
            o3 = current["ozone"]


            # -------------------------------------------------
            # Step 3: Prepare ML input
            # -------------------------------------------------

            features = [[
                pm25,
                pm10,
                no2,
                so2,
                o3
            ]]


            # -------------------------------------------------
            # Step 4: Random Forest Prediction
            # -------------------------------------------------

            if model is not None:

                prediction = model.predict(features)

                air_quality_status = str(prediction[0])

            else:

                air_quality_status = "Model not loaded"


            # -------------------------------------------------
            # Step 5: Prepare final response
            # -------------------------------------------------

            air_quality = {

                "location": "Kochi",

                "time": current["time"],

                "PM2.5": pm25,

                "PM10": pm10,

                "NO2": no2,

                "SO2": so2,

                "O3": o3,

                "air_quality_status": air_quality_status
            }


            # -------------------------------------------------
            # Display result on server
            # -------------------------------------------------

            print("\nLive Air Quality Data:")
            print(json.dumps(air_quality, indent=4))


            # -------------------------------------------------
            # Send response to CoAP client
            # -------------------------------------------------

            payload = json.dumps(air_quality)

            return Message(
                payload=payload.encode("utf-8"),
                content_format=50
            )


        except Exception as e:

            error = {
                "error": str(e)
            }

            print("\nError:", str(e))

            return Message(
                payload=json.dumps(error).encode("utf-8"),
                content_format=50
            )


# =========================================================
# Main CoAP Server
# =========================================================

async def main():

    # Create CoAP site
    root = resource.Site()


    # Add /airquality resource
    root.add_resource(
        ["airquality"],
        AirQualityResource()
    )


    # Start CoAP server
    await Context.create_server_context(
        root,
        bind=("127.0.0.1", 5683)
    )


    print("\n---------------------------------------")
    print("       CoAP Air Quality Server")
    print("---------------------------------------")
    print("Server   : coap://127.0.0.1:5683")
    print("Resource : /airquality")
    print("ML Model : Random Forest")
    print("Features : PM2.5, PM10, NO2, SO2, O3")
    print("---------------------------------------")
    print("Waiting for GET request...")
    print("---------------------------------------")


    # Keep server running
    await asyncio.get_running_loop().create_future()


# =========================================================
# Run Server
# =========================================================

if __name__ == "__main__":
    asyncio.run(main())