FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY scripts ./scripts
COPY train.py verify_pipeline.py pytest.ini ./
COPY trading_model_sniper_v5.pkl ./trading_model_sniper_v5.pkl

RUN python scripts/setup_model.py || true

EXPOSE 8000 7860

ENV ENABLE_INFERENCE_MLFLOW=false
ENV SENTIMENT_BACKEND=vader

CMD ["sh", "scripts/start.sh"]
