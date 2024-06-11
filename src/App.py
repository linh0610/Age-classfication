import cv2
import numpy as np
import torch
from PIL import Image, ImageEnhance
from torchvision import transforms
from model import AgeModel, device


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
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video stream
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame from BGR to RGB (OpenCV uses BGR by default)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert the frame to a PIL image
    pil_image = Image.fromarray(frame_rgb)

    # Enhance the image (optional, you can adjust the factors as needed)
    # enhancer = ImageEnhance.Contrast(pil_image)
    # pil_image = enhancer.enhance(2)  # Increase contrast

    # enhancer = ImageEnhance.Brightness(pil_image)
    # pil_image = enhancer.enhance(1.5)  # Increase brightness

    # Preprocess the frame
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        predicted_age_index = torch.argmax(output).item()
        predicted_age = classes[predicted_age_index]

    # Display the frame with predicted age
    frame_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV
    cv2.putText(frame_bgr, f"Predicted Age: {predicted_age}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Real-Time Age Tracking', frame_bgr)

    # Check for exit command by pressing 'q' or clicking the 'X' button
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.getWindowProperty('Real-Time Age Tracking', cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()