FROM --platform=linux/arm64 python:3.11-slim

LABEL org.opencontainers.image.source="https://github.com/vivek5200/sama"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/cache/huggingface

WORKDIR /app

COPY spaces/sama-itsm-demo/requirements.txt ./requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY sama ./sama
COPY spaces/sama-itsm-demo/app.py spaces/sama-itsm-demo/choice_head_5class.pt ./

EXPOSE 7860

CMD ["python", "app.py"]
