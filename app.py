from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model & scaler
with open("lr2.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        usd_inr = float(request.form["usd_inr"])
        usd_inr_scaled = scaler.transform(np.array([[usd_inr]]))
        prediction = model.predict(usd_inr_scaled)[0]
        return render_template("index.html", prediction=round(prediction, 2))
    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)
