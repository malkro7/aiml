# producer/Dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip \
    && pip install pandas kafka-python scikit-learn mlflow

CMD ["python", "producer.py"]