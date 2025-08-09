# client1/Dockerfile (shared by all clients)
FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip \
    && pip install flwr tensorflow pandas scikit-learn kafka-python matplotlib seaborn mlflow