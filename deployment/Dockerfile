FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /app/data/input /app/data/output/results /app/data/output/figures

EXPOSE 80

ENV NAME MedExtract

CMD ["python", "medextract.py", "--config", "config/config.yaml"]