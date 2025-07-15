import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
import warnings 
warnings.filterwarnings('ignore')


st.title("🏏 IPL Score Predictor")
st.image("image.png", width=100)


df = pd.read_csv("ipl_data.csv")


label_encoders = {}
categorical_cols = ['bat_team', 'bowl_team', 'venue', 'batsman', 'bowler']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

scaler = joblib.load("scaler.pkl")

from keras.losses import Huber
model = load_model("ipl_score_prediction_model.h5", custom_objects={"Huber": Huber}) 


venue = st.selectbox("Select Venue", label_encoders['venue'].classes_)
batting_team = st.selectbox("Select Batting Team", label_encoders['bat_team'].classes_)
bowling_team = st.selectbox("Select Bowling Team", label_encoders['bowl_team'].classes_)
striker = st.selectbox("Select Striker", label_encoders['batsman'].classes_)
bowler = st.selectbox("Select Bowler", label_encoders['bowler'].classes_)

runs = st.number_input("Current Runs", min_value=0)
wickets = st.number_input("Current Wickets", min_value=0, max_value=10)
overs = st.number_input("Current Overs", min_value=0.0, max_value=20.0, step=0.1)
striker_ind = st.number_input("Striker Index (0 or 1)", min_value=0, max_value=1)


if st.button("Predict Score"):
    input_features = [
        label_encoders['bat_team'].transform([batting_team])[0],
        label_encoders['bowl_team'].transform([bowling_team])[0],
        label_encoders['venue'].transform([venue])[0],
        runs,
        wickets,
        overs,
        striker_ind,
        label_encoders['batsman'].transform([striker])[0],
        label_encoders['bowler'].transform([bowler])[0],
    ]

    input_array = np.array(input_features).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)

    # Predict
    prediction = model.predict(input_array_scaled)
    predicted_score = int(prediction[0])

    st.success(f"🎯 Predicted Total Runs: {predicted_score}")
