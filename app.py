import gradio as gr
import pickle
import numpy as np

# Load trained model and scaler
with open("lr2.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Define Prediction Function
def predict_gold_price(usd_inr):
    usd_inr_scaled = scaler.transform(np.array([[usd_inr]]))
    prediction = model.predict(usd_inr_scaled)[0]
    return f"Predicted Gold Price: ₹{prediction:.2f}"

# Gradio Interface
iface = gr.Interface(
    fn=predict_gold_price,
    inputs=gr.Number(label="💵 USD/INR Exchange Rate"),
    outputs=gr.Textbox(label="📊 Gold Price Prediction"),
    title="Gold Price Prediction 💰",
    description="Enter the USD/INR exchange rate and get the predicted gold price.",
    theme="compact"
)

# Launch App
if __name__ == "__main__":
    iface.launch()
