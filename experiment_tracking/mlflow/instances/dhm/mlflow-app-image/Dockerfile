FROM ghcr.io/mlflow/mlflow:v2.11.0

RUN apt-get update && apt-get install -y curl && apt-get clean
RUN pip install --no-cache-dir psycopg2-binary cryptography pymysql boto3

EXPOSE 5000

