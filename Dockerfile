# Container der evaluerer den færdigtrænede model på et lille udsnit af testsættet.
FROM python:3.13-slim

WORKDIR /app
ENV MPLBACKEND=Agg \
    PYTHONUNBUFFERED=1

# CPU-udgaven af PyTorch, så imaget ikke bliver flere GB stort (CUDA-udgaven er meget større)
RUN pip install --no-cache-dir torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kode og trænede modeller
COPY train.py test.py split_data.py image_sizes.py ./
COPY checkpoints/ checkpoints/

# Eksempelbillederne lægges der, hvor test.py forventer testdata (data/test/fresh, data/test/rotten)
COPY sample_data/ data/test/

CMD ["python", "test.py"]
