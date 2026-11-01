FROM python:3.10-slim

WORKDIR /code
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/code/.hf_cache

# CPU-only PyTorch (much smaller than the default build). PyPI stays available for its dependencies.
RUN pip install --no-cache-dir "torch==2.14.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Bake the two pretrained models into the image so the container starts without internet
RUN python -c "from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM; \
a='distilbert-base-cased-distilled-squad'; b='google/flan-t5-small'; \
AutoTokenizer.from_pretrained(a); AutoModelForQuestionAnswering.from_pretrained(a); \
AutoTokenizer.from_pretrained(b); AutoModelForSeq2SeqLM.from_pretrained(b)"

COPY src ./src
COPY app ./app
COPY models/risk_classifier ./models/risk_classifier
COPY data/*.txt ./data/

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]