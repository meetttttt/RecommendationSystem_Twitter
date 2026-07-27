FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.13.0+cpu \
    && grep -v "^torch==" requirements.txt > /tmp/requirements-without-torch.txt \
    && pip install --no-cache-dir -r /tmp/requirements-without-torch.txt

COPY src ./src
COPY load_tests ./load_tests
COPY data/processed_data.csv ./data/processed_data.csv
COPY models/neural_ranker.pt ./models/neural_ranker.pt

EXPOSE 8000 8089

CMD ["uvicorn", "floter_ai.api:app", "--host", "0.0.0.0", "--port", "8000"]
