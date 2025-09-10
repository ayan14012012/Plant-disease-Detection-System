import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import mobilenet_v2
from PIL import Image
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 39  # Updated to 39 classes

# Replace this list with your actual 39 class names in the exact order
class_names = [
    "Apple Apple scab",
    "Apple Black rot",
    "Apple Cedar apple rust",
    "Apple healthy",
    "Background without leaves",
    "Blueberry healthy",
    "Cherry Powdery mildew",
    "Cherry healthy",
    "Corn Cercospora leaf spot Gray leaf spot",
    "Corn Common rust",
    "Corn healthy",
    "Corn Northern Leaf Blight",
    "Grape Black rot",
    "Grape Esca (Black Measles)",
    "Grape Leaf blight (Isariopsis Leaf Spot)",
    "Grape healthy",
    "Orange Haunglongbing (Citrus greening)",
    "Peach Bacterial spot",
    "Peach healthy",
    "Pepper bell Bacterial spot",
    "Pepper bell healthy",
    "Potato Early blight",
    "Potato Late blight",
    "Potato healthy",
    "Raspberry healthy",
    "Soybean healthy",
    "Squash Powdery mildew",
    "Strawberry Leaf scorch",
    "Strawberry healthy",
    "Tomato Bacterial spot",
    "Tomato Early blight",
    "Tomato Late blight",
    "Tomato Leaf Mold",
    "Tomato Septoria leaf spot",
    "Tomato Spider mites Two-spotted spider mite",
    "Tomato Target Spot",
    "Tomato Tomato mosaic virus",
    "Tomato Tomato Yellow Leaf Curl Virus",
    "Tomato healthy"
]



# Build the model architecture without pretrained weights
model = mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
model = model.to(device)

# Load the saved weights
model_path = r"C:\Users\Vashi\Downloads\plant_disease_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Image preprocessing transforms
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

def predict_frame(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    img_t = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_t)
        _, predicted = torch.max(output, 1)
        idx = predicted.item()
        if idx >= len(class_names):
            return "Unknown"
        return class_names[idx]

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting webcam. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        label = predict_frame(frame)
        cv2.putText(frame, f"Prediction: {label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1)

        cv2.imshow('Plant Disease Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
