import streamlit as st
import pandas as pd
import pickle

# Load the trained pipeline
pipe = pickle.load(open('XGBRegressionModel.pkl', 'rb'))

# Define team options
teams = [
    'Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
    'Delhi Daredevils', 'Sunrisers Hyderabad'
]

# Streamlit App Title
st.title("IPL Score Prediction")

# Input Fields
st.header("Input Match Details")
batting_team = st.selectbox("Select Batting Team", teams)
bowling_team = st.selectbox("Select Bowling Team", [team for team in teams if team != batting_team])
overs = st.number_input("Overs Completed", min_value=5.0, max_value=20.0, step=0.1)
runs = st.number_input("Current Runs", min_value=0, step=1)
wickets = st.number_input("Wickets Fallen", min_value=0, max_value=10, step=1)
runs_last_5 = st.number_input("Runs Scored in Last 5 Overs", min_value=0, step=1)
wickets_last_5 = st.number_input("Wickets Fallen in Last 5 Overs", min_value=0, max_value=10, step=1)

# Prediction Button
if st.button("Predict Total Score"):
    # Prepare input DataFrame
    input_data = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'overs': [overs],
        'runs': [runs],
        'wickets': [wickets],
        'runs_last_5': [runs_last_5],
        'wickets_last_5': [wickets_last_5]
    })
    
    # Make prediction
    prediction = pipe.predict(input_data)[0]
    st.subheader(f"Predicted Total Runs: {prediction:.2f}")
