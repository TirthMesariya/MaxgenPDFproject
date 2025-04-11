FROM python:3.10-slim

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

COPY . /app

COPY model_3.pth /app/myapp/model_3.pth

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 1234

CMD ["sh", "-c", "python manage.py migrate && python manage.py runserver 0.0.0.0:1234"]