import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# load trained model
model = pickle.load(open("car_price_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    # get human-friendly values from form
    present_price = float(request.form["Present_Price"])
    kms_driven    = float(request.form["Kms_Driven"])
    owner         = int(request.form["Owner"])
    year          = int(request.form["Year"])
    fuel_type     = request.form["Fuel_Type"]       # "Petrol", "Diesel", or "CNG"
    seller_type   = request.form["Seller_Type"]     # "Dealer" or "Individual"
    transmission  = request.form["Transmission"]    # "Automatic" or "Manual"

    # --- encode exactly as the notebook did ---

    # Car_Age: notebook used 2026 - Year
    car_age = 2026 - year

    # get_dummies(drop_first=True) keeps Diesel & Petrol, drops CNG (the "first" alphabetically)
    fuel_diesel = 1 if fuel_type == "Diesel" else 0
    fuel_petrol = 1 if fuel_type == "Petrol" else 0

    # get_dummies(drop_first=True) keeps Individual, drops Dealer
    seller_individual = 1 if seller_type == "Individual" else 0

    # get_dummies(drop_first=True) keeps Manual, drops Automatic
    transmission_manual = 1 if transmission == "Manual" else 0

    # create dataframe with correct feature names and ORDER
    features = pd.DataFrame([[
        present_price,
        kms_driven,
        owner,
        car_age,
        fuel_diesel,
        fuel_petrol,
        seller_individual,
        transmission_manual
    ]], columns=[
        "Present_Price",
        "Kms_Driven",
        "Owner",
        "Car_Age",
        "Fuel_Type_Diesel",
        "Fuel_Type_Petrol",
        "Seller_Type_Individual",
        "Transmission_Manual"
    ])

    # make prediction
    prediction = model.predict(features)
    output = round(prediction[0], 2)


    if present_price > 20:
        warning = "⚠️ Warning: This model is trained on budget cars (under ₹20L). Prediction may not be accurate."
    else:
        warning = ""

    return render_template("index.html", prediction_text=f"₹{output} Lakhs", warning=warning)

if __name__ == "__main__":
    app.run(debug=True)