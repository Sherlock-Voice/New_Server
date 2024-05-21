# backend/Dockerfile
FROM python:3.9

COPY . /app
WORKDIR /app

COPY requirements.txt .

RUN apt-get update && apt-get install -y sqlite3
RUN pip install numpy
RUN pip install scikit-learn
RUN apt-get update && apt-get install -y libsndfile1
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY models/shuffle_400.csv models/shuffle_400.csv

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]