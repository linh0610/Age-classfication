from flask import Flask, request, render_template, jsonify
import torch
from torchvision import transforms
from PIL import Image
import io
from model import AgeModel, device

app = Flask(__name__)

model = AgeModel(num_classes=14)
model = torch.load('C:/Users/linh0/best_model.pth', map_location=device)
model.eval()

classes = [
    '1-2', '3-5', '6-10', '11-14', '15-17',
    '18-22', '23-27', '28-33', '34-40', '41-50',
    '51-60', '61-70', '71-80', 'over80'
]
mean = [ 0.0739, -0.0482, -0.1140]
std = [0.4999, 0.4792, 0.4785]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
])


def classify_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs.data, 1)
        predicted_class = classes[predicted.item()]

    return predicted_class


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            img_bytes = file.read()
            prediction = classify_image(img_bytes)
            return jsonify({'class': prediction})


if __name__ == '__main__':
    app.run(debug=True)