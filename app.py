from flask import Flask, render_template, request, jsonify
import os
import json
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import requests
from werkzeug.utils import secure_filename
import cohere
from deep_translator import GoogleTranslator

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

co = cohere.Client("ZybYuuzQq3KuS7pGWkgfIZi72s4ddVCGX2896F0z")  # Your Cohere API key

# Load treatment data
with open('treatment_data.json', 'r', encoding='utf-8') as f:
    treatment_data = json.load(f)

# Class labels
class_names = [
    "Apple Apple scab", "Apple Black rot", "Apple Cedar apple rust", "Apple healthy",
    "Background without leaves", "Blueberry healthy", "Cherry Powdery mildew", "Cherry healthy",
    "Corn Cercospora leaf spot Gray leaf spot", "Corn Common rust", "Corn healthy",
    "Corn Northern Leaf Blight", "Grape Black rot", "Grape Esca (Black Measles)",
    "Grape Leaf blight (Isariopsis Leaf Spot)", "Grape healthy",
    "Orange Haunglongbing (Citrus greening)", "Peach Bacterial spot", "Peach healthy",
    "Pepper bell Bacterial spot", "Pepper bell healthy", "Potato Early blight", "Potato Late blight",
    "Potato healthy", "Raspberry healthy", "Soybean healthy", "Squash Powdery mildew",
    "Strawberry Leaf scorch", "Strawberry healthy", "Tomato Bacterial spot", "Tomato Early blight",
    "Tomato Late blight", "Tomato Leaf Mold", "Tomato Septoria leaf spot",
    "Tomato Spider mites Two-spotted spider mite", "Tomato Target Spot",
    "Tomato Tomato mosaic virus", "Tomato Tomato Yellow Leaf Curl Virus", "Tomato healthy"
]

# Load model
NUM_CLASSES = len(class_names)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model.load_state_dict(torch.load("plant_disease_model.pth", map_location=device))
model.eval().to(device)

transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        return class_names[predicted.item()]

def get_weather():
    try:
        api_key = 'ddd01d64999949ae8f3ae405c8662ac0'
        lat, lon = 28.7041, 77.1025
        current_url = f'https://api.weatherbit.io/v2.0/current?lat={lat}&lon={lon}&key={api_key}'
        forecast_url = f'https://api.weatherbit.io/v2.0/forecast/daily?lat={lat}&lon={lon}&days=3&key={api_key}'

        current = requests.get(current_url).json()
        forecast = requests.get(forecast_url).json()

        weather = {
            'current': {
                'city': current['data'][0]['city_name'],
                'temperature': current['data'][0]['temp'],
                'description': current['data'][0]['weather']['description'],
                'humidity': current['data'][0]['rh'],
                'icon': current['data'][0]['weather']['icon']
            },
            'forecast': [{
                'date': day['datetime'],
                'description': day['weather']['description'],
                'temp': day['temp'],
                'rain_probability': day.get('pop', 0)
            } for day in forecast['data']]
        }
        return weather
    except Exception as e:
        print("Weather API Error:", e)
        return {'error': 'Weather not available'}

def get_market_prices():
    try:
        url = "https://api.data.gov.in/resource/9ef84268-d588-465a-a308-a864a43d0070"
        params = {
            'api-key': '579b464db66ec23bdd00000187c37d2065f844114d8f3ff5c092be6e',
            'format': 'json',
            'limit': 10
        }
        response = requests.get(url, params=params).json()
        records = response.get("records", [])

        return [{
            'crop': r.get('commodity', 'N/A'),
            'market': r.get('market', 'N/A'),
            'price': r.get('modal_price', 'N/A')
        } for r in records]
    except Exception as e:
        print("Market Price API Error:", e)
        return []

# Translation helpers
def translate(text, source, target):
    try:
        return GoogleTranslator(source=source, target=target).translate(text)
    except Exception as e:
        print(f"Translation error from {source} to {target}:", e)
        return text

def generate_ai_reply(user_input):
    try:
        response = co.generate(
            model='command-r-plus',
            prompt=f"You are an expert farming assistant. Answer this user query clearly:\n\nUser: {user_input}\nAI:",
            max_tokens=200,
            temperature=0.7
        )
        return response.generations[0].text.strip()
    except Exception as e:
        print("Cohere error:", e)
        return "Sorry, I'm having trouble answering right now."

@app.route("/chatbot", methods=["POST"])
def chatbot():
    data = request.get_json()
    user_input = data.get("message", "")
    lang = data.get("lang", "en")

    try:
        if lang != "en":
            user_input = translate(user_input, source=lang, target="en")

        response = generate_ai_reply(user_input)

        if lang != "en":
            response = translate(response, source="en", target=lang)

        return jsonify({'reply': response})
    except Exception as e:
        print("Chatbot error:", e)
        return jsonify({'reply': "Sorry, I'm having trouble answering right now."})

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = treatment = image_path = None
    lang = request.form.get("lang", "en")
    weather = get_weather()
    market_prices = get_market_prices()

    if request.method == "POST" and 'file' in request.files:
        file = request.files['file']
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            image_path = 'uploads/' + filename
            prediction = predict_image(filepath)

            treatment_entry = treatment_data.get(prediction, {})
            treatment = treatment_entry.get(lang, "Treatment info not available.") if isinstance(treatment_entry, dict) else treatment_entry
            if isinstance(treatment, list):
                treatment = "<br>".join(treatment)

    return render_template("index.html",
                       image_path=image_path,
                       prediction=prediction,
                       treatment=treatment,
                       weather=weather,
                       market_prices=market_prices,
                       lang=lang)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
