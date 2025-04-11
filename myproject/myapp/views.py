from django.shortcuts import render  
from rest_framework.views import APIView  
from rest_framework.response import Response  
from rest_framework import status  


from django.shortcuts import get_object_or_404  

    








# ================================
from django.shortcuts import render, redirect
from .models import *


def index(request):
    return render(request, 'index.html')


def add_company(request):
    if request.method == "POST":
        name = request.POST.get('name')
        contact = request.POST.get('contact')
        email = request.POST.get('email')
        password = request.POST.get('password')
        contact_person = request.POST.get('contact_person')
        address = request.POST.get('address')

   
        if Company.objects.filter(email=email).exists():
            return render(request, 'add_company.html', {'error': 'Email already registered!'})

        # If the company does not exist, create it
        Company.objects.create(
            name=name,
            contact=contact,
            email=email,
            password=password,  # Store securely in real-world scenarios
            contact_person=contact_person,
            address=address
        )

        return render(request, 'add_company.html')

    return render(request, 'add_company.html')




def company_list(request):
    companies = Company.objects.all()  # Fetch all company records
    return render(request, 'company_list.html', {'companies': companies})



# ==================



def add_CompanyDetails(request):
    companies = Company.objects.all()  
     

    if request.method == "POST":
        company_id = request.POST.get("company")
        files = request.FILES.getlist("file")  # Supports multiple file uploads

        if company_id:
            company = Company.objects.get(id=company_id)

            # Loop through all uploaded files and save them
            for file in files:
                CompanyDetails.objects.create(company=company, file=file)
            
            return render(request, 'add.html')

    return render(request, 'add.html', {'companies': companies })


def add_list(request):
    company_details = CompanyDetails.objects.all()  # Fetch all company records
    return render(request, 'add_list.html', {'company_details': company_details})
from django.http import JsonResponse
import requests
def company_detail123(request, detail_id):
    company_detail = CompanyDetails.objects.get(id=detail_id)

    
    api_url = "http://127.0.0.1:8000/api/"

    
    files = {'file': company_detail.file.open('rb')} if company_detail.file else None

    try:
        
        response = requests.post(api_url, files=files)
        response_data = response.json()
        print(response_data)
        return render(request, 'company_details.html', {
            "employees": response_data['extracted_text'],  # Pass data to template
        })

    except requests.exceptions.RequestException as e:
        return JsonResponse({"error": str(e)}, status=500)


    # Render response in template
    return render(request, 'company_details.html', {
        "employees": response_data,
    })

# ================= api ===================




import torch
import torchvision.transforms as T
import torchvision
import pytesseract
import cv2
import numpy as np
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.parsers import MultiPartParser, FormParser
from PIL import Image
import io

# Load the trained Faster R-CNN model
model_path = r"C:\Users\Tirth\Downloads\MaxgenPDFproject\myproject\model_3.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABEL_MAP = {
    1: "Name",
    2: "CODE",
    3: "Gross_Salary",
    4: "I.TAX",
    5: "Profit_TAX",
    6: "Treasure_voucher_No",
    7: "Treasure_voucher_Date",
    8: "DDO"
}

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = len(LABEL_MAP) + 1
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def extract_text_from_box(image, box):
    pytesseract.pytesseract.tesseract_cmd = r"C:\Users\Tirth\Downloads\MaxgenPDFproject\myproject\Tesseract-OCR\tesseract.exe"
    x1, y1, x2, y2 = map(int, box)
    cropped_region = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    extracted_text = pytesseract.image_to_string(binary, config='--psm 6').strip()
    return extracted_text if extracted_text else "N/A"

def detect_text(image):
    image_cv = np.array(image)
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image_tensor)[0]

    extracted_data = {label: [] for label in LABEL_MAP.values()}
    boxes = predictions["boxes"].cpu().numpy()
    labels = predictions["labels"].cpu().numpy()
    scores = predictions["scores"].cpu().numpy()
    
    sorted_boxes_indices = np.argsort(boxes[:, 0])
    boxes = boxes[sorted_boxes_indices]
    labels = labels[sorted_boxes_indices]
    scores = scores[sorted_boxes_indices]

    treasure_voucher_no = treasure_voucher_date = ddo = None
    for i in range(len(boxes)):
        if scores[i] > 0.5:
            box = boxes[i]
            label_id = labels[i]
            label_name = LABEL_MAP.get(label_id, "Unknown")
            extracted_text = extract_text_from_box(image_cv, box)
            if extracted_text:
                extracted_data[label_name].append(extracted_text)
                if label_name == "Treasure_voucher_No":
                    treasure_voucher_no = extracted_text
                if label_name == "Treasure_voucher_Date":
                    treasure_voucher_date = extracted_text
                if label_name == "DDO":
                    ddo = extracted_text
    
    max_rows = max(len(v) for v in extracted_data.values() if isinstance(v, list))
    for key in extracted_data:
        if isinstance(extracted_data[key], list):
            while len(extracted_data[key]) < max_rows:
                extracted_data[key].append(np.nan)
    
    if treasure_voucher_no:
        extracted_data["Treasure_voucher_No"] = [treasure_voucher_no] * max_rows
    if treasure_voucher_date:
        extracted_data["Treasure_voucher_Date"] = [treasure_voucher_date] * max_rows
    if ddo:
        extracted_data["DDO"] = [ddo] * max_rows

    return extracted_data




import pdf2image  
import os
import io
import pandas as pd
import numpy as np
from PIL import Image
from rest_framework.response import Response
from rest_framework.views import APIView
import re
import zipfile

class PredictAPIView(APIView):

    def post(self, request, *args, **kwargs):
        if 'file' in request.FILES:
            file = request.FILES['file']
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
            output = detect_text(self,image)
            print(output)
            df = pd.DataFrame(output).replace({np.nan: None})
            print(df)
            json_output = df.to_dict(orient="records")
            return Response({"extracted_text": json_output})
        return Response({"error": "No file uploaded"}, status=400)
    
    

#   https://www.python.org/ftp/python/3.1.5/Python-3.1.5.tgz       

from django.core.files.storage import FileSystemStorage
from django.http import JsonResponse, HttpResponse
from django.views import View
from django.conf import settings
from django.urls import path
from django.shortcuts import render
import zipfile
import os
import torch
import torchvision.transforms as T
import pandas as pd
from PIL import Image
import numpy as np
from pdf2image import convert_from_path
import cv2
import pytesseract
import torchvision

# Configure Tesseract
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Set Poppler path manually
poppler_path = r"C:\Users\Tirth\Downloads\MaxgenPDFproject\myproject\bin"

# Model path
model_path = r"C:\Users\Tirth\Downloads\MaxgenPDFproject\myproject\model_3.pth"

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
num_classes = len(LABEL_MAP) + 1
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# PDF and Image Processing Functions
def convert_pdf_to_high_res_images(pdf_path, dpi=400, target_size=(4958, 7016)):
    images = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)
    return [img.resize(target_size, Image.LANCZOS) for img in images]


def extract_text_from_box(image, box):
    x1, y1, x2, y2 = map(int, box)
    cropped_region = image[y1:y2, x1:x2]
    gray = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return pytesseract.image_to_string(binary, config='--psm 6').strip() or "N/A"


def detect_text(image, model):
    image_cv = np.array(image)
    image_tensor = T.ToTensor()(image).unsqueeze(0).to(device)
    with torch.no_grad():
        predictions = model(image_tensor)[0]

    extracted_data = {label: [] for label in LABEL_MAP.values()}
    boxes, labels, scores = predictions['boxes'].cpu().numpy(), predictions['labels'].cpu().numpy(), predictions['scores'].cpu().numpy()
    sorted_indices = np.lexsort((boxes[:, 0], boxes[:, 1]))
    boxes, labels, scores = boxes[sorted_indices], labels[sorted_indices], scores[sorted_indices]

    for i in range(len(boxes)):
        if scores[i] > 0.4:
            label_name = LABEL_MAP.get(labels[i], "Unknown")
            extracted_text = extract_text_from_box(image_cv, boxes[i])
            if extracted_text:
                extracted_data[label_name].append(extracted_text)

    return extracted_data





class UploadZipView(View):
    def get(self, request):
        return render(request, 'upload.html')

    def post(self, request):
        uploaded_file = request.FILES['file']
        fs = FileSystemStorage()
        zip_path = fs.save(uploaded_file.name, uploaded_file)
        zip_full_path = os.path.join(settings.MEDIA_ROOT, zip_path)

        extracted_data = {}
        with zipfile.ZipFile(zip_full_path, 'r') as zip_ref:
            zip_ref.extractall(settings.MEDIA_ROOT)
            for file_name in zip_ref.namelist():
                if file_name.lower().endswith('.pdf'):
                    pdf_path = os.path.join(settings.MEDIA_ROOT, file_name)
                    images = convert_pdf_to_high_res_images(pdf_path)
                    file_data = {label: [] for label in LABEL_MAP.values()}
                    for image in images:
                        page_data = detect_text(image, model)
                        max_length = max(len(values) for values in page_data.values())
                        for key in page_data.keys():
                            page_data[key].extend(["N/A"] * (max_length - len(page_data[key])))
                        for i in range(max_length):
                            for key, values in page_data.items():
                                file_data[key].append(values[i])
                    extracted_data[file_name] = file_data

        fs.delete(zip_path)
        return render(request, 'display_data.html', {'extracted_data': extracted_data})

 
    