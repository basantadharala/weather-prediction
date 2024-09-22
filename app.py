import pandas as pd
import streamlit as st
import pickle
page_bg_img = '''
<style>
.main{
    background-image: url("https://images.unsplash.com/photo-1580193769210-b8d1c049a7d9?q=80&w=2948&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
    opacity:1;
}

</style>
'''
st.markdown(page_bg_img,unsafe_allow_html=True)

try:
    with open('weather.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open('label_encoder.pkl', 'rb') as le_file:
        le = pickle.load(le_file)
except Exception as e:
    st.write("Error loading model, scaler, or label encoder:", e)
    st.stop()



st.title("Weather Prediction App")

st.write("Enter the precipitation, max temperature, min temperature, and wind to predict the weather type.")

# Input sliders for user input
precipitation = st.slider("Precipitation", min_value=0.0, max_value=100.0, value=0.1)
temp_max = st.slider("Maximum Temperature", min_value=-5.0, max_value=60.0, value=15.0)
temp_min = st.slider("Minimum Temperature", min_value=-10.0, max_value=30.0, value=7.0)
wind = st.slider("Wind", min_value=0.0, max_value=21.0, value=5.0)

input_data = pd.DataFrame([[precipitation,temp_max,temp_min,wind]])


try:
    input_data_scaled = scaler.transform(input_data)  # Debugging output
except Exception as e:
    st.write("Error scaling input data:", e)
    st.stop()

# Make a prediction when the button is pressed
if st.button("Predict Weather"):
    try:
        prediction = model.predict(input_data_scaled)
        predicted_label = le.inverse_transform(prediction)[0]
        st.write("The predicted weather is:", predicted_label)
    except Exception as e:
        st.write("Error making prediction:", e)
