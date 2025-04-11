import torch
import torchvision.transforms as T
import cv2
import pytesseract
import torchvision
import pandas as pd
from PIL import Image
import numpy as np
from pdf2image import convert_from_path
import os

# Set Poppler path manually (Update this path as needed)
poppler_path = r"C:\Users\Shree\Desktop\myproject (1)\myproject\bin"

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Update with your Tesseract path
# Load the trained Faster R-CNN model
model_path =  r"C:\Users\Shree\Desktop\ML\model_3.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define label mapping
LABEL_MAP = {
    1: "Name",
    2: "CODE",
    3: "Gross Salary",
    4: "I.TAX",
    5: "Profit TAX",
    6: "Treasure voucher No.",
    7: "Treasure voucher Date",
    8: "DDO"
}

# Load pre-trained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = len(LABEL_MAP) + 1  # Add 1 for background class
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

# Load trained weights
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def convert_pdf_to_high_res_images(pdf_path, dpi=400, target_size=(4958, 7016)):
    images = convert_from_path(pdf_path, dpi=dpi, fmt="png")
    resized_images = [img.resize(target_size, Image.LANCZOS) for img in images]
    return resized_images

def extract_text_from_box(image, box):
    x1, y1, x2, y2 = map(int, box)
    cropped_region = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    extracted_text = pytesseract.image_to_string(binary, config='--psm 6').strip()
    return extracted_text if extracted_text else "N/A"

def detect_text(image, model):
    image_cv = np.array(image)
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    extracted_data = {label: [] for label in LABEL_MAP.values()}
    boxes = predictions["boxes"].cpu().numpy()
    labels = predictions["labels"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()
    sorted_indices = np.argsort(boxes[:, 0])
    boxes, labels, scores = boxes[sorted_indices], labels[sorted_indices], scores[sorted_indices]

    for i in range(len(boxes)):
        if scores[i] > 0.4:
            box, label_id = boxes[i], labels[i]
            label_name = LABEL_MAP.get(label_id, "Unknown")
            extracted_text = extract_text_from_box(image_cv, box)
            
            if extracted_text:
                extracted_data[label_name].append(extracted_text)
    
    max_rows = max(len(v) for v in extracted_data.values() if isinstance(v, list))
    for key in extracted_data:
        extracted_data[key] += [np.nan] * (max_rows - len(extracted_data[key]))
    
    return extracted_data

def process_pdf(pdf_path, model):
    images = convert_pdf_to_high_res_images(pdf_path)
    all_data = []
    
    for i, image in enumerate(images):
        print(f"Processing {os.path.basename(pdf_path)} - Page {i+1}/{len(images)}...")
        extracted_data = detect_text(image, model)
        df = pd.DataFrame(extracted_data)
        print(df)  # Display the DataFrame
        all_data.append(df)
    
    final_df = pd.concat(all_data, ignore_index=True)
    print(final_df)  # Display the final DataFrame

def process_pdf_folder(input_folder, model):
    pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".pdf")]
    
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_folder, pdf_file)
        process_pdf(pdf_path, model)
    
    print("All PDFs processed successfully!")

input_pdf_folder = r"C:\Users\Shree\Desktop\ML\New folder\New folder"
process_pdf_folder(input_pdf_folder, model)
