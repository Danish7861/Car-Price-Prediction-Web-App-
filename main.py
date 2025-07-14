from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import pickle
import numpy as np  # Add at top


app = FastAPI(title="Car Price Prediction API 🚗")

# Load the model
with open("car_price_model.pkl", "rb") as f:
    model = pickle.load(f)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def form_ui(request: Request):
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict/", response_class=HTMLResponse)
async def predict_ui(
    request: Request,
    Location: str = Form(...),
    Fuel_Type: str = Form(...),
    Transmission: str = Form(...),
    Brand: str = Form(...),
    Owner_Type: str = Form(...),
    Year: int = Form(...),
    Kilometers_Driven: int = Form(...),
    Mileage: float = Form(...),
    Engine: float = Form(...),
    Power: float = Form(...),
    Seats: float = Form(...)
):
    input_dict = {
        "Location": Location,
        "Fuel_Type": Fuel_Type,
        "Transmission": Transmission,
        "Brand": Brand,
        "Owner_Type": Owner_Type,
        "Year": Year,
        "Kilometers_Driven": Kilometers_Driven,
        "Mileage": Mileage,
        "Engine": Engine,
        "Power": Power,
        "Seats": Seats,
        "New_Price": 0,                # Default/fake value
        "New_Price_Missing": 1        # Pretend it’s missing
    }
    input_df = pd.DataFrame([input_dict])
    log_price = model.predict(input_df)[0]
    prediction = np.expm1(log_price)*331223.60  # if you used np.log1p during training

    return templates.TemplateResponse("form.html", {
        "request": request,
        "prediction": f"Predicted Car Price: PKR{prediction:,.2f}"
    })
