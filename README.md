# Carbon-Footprint-Estimator

![Image](https://github.com/user-attachments/assets/7007d99e-a726-44f4-94ca-9ed480ae521b)

# 🌍 Carbon Emission Predictor

A web-based application that predicts carbon dioxide (CO₂) emissions based on transportation details like distance, fuel consumption, speed, cargo weight, and traffic conditions. Users can enter **source and destination** or **manual trip details** to get accurate emission estimates.

![Flask App Screenshot](https://user-images.githubusercontent.com/your-image.png) <!-- Add your screenshot here -->

---

## 🚀 Features

- 🌐 Location-based prediction using source & destination
- ✍️ Manual data entry option (distance, fuel, speed, weight, traffic)
- 📊 Predicts estimated CO₂ emissions instantly
- 🧠 Built with Flask, Machine Learning & Bootstrap UI
- ✅ Responsive and aesthetic user interface

---

## 🧠 How It Works

1. **Manual inputs** allow for fine-tuned control of vehicle parameters.
2. A **machine learning model** (e.g., regression) predicts CO₂ emissions based on inputs.
3. Prediction is displayed in a user-friendly format.

---

## 📁 Project Structure

```bash
carbon_emission_predictor/
├── static/
│   └── css/
│       └── styles.css
├── templates/
│   └── index.html
├── model/
│   └── emission_model.pth       # Trained ML model
├── app.py                        # Main Flask backend
├── requirements.txt              # Python dependencies
├── README.md

```


### 🔧 Tech Stack
- Python 3
 
- PyTorch

- scikit-learn

- Flask

- HTML/CSS

### 🧪 Example Usage
- Enter:

Source: Bangalore

Destination: Mysore

- OR

Distance: 145

Speed: 60

Fuel Consumption: 8

Cargo Weight: 1000

Traffic: Medium

Click on "Predict Emission"

### OUTPUT

Estimated Carbon Emission: 24.56 kg CO₂

