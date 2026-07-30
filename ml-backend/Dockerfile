FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (psycopg needs gcc)
RUN apt-get update && \
    apt-get install -y build-essential cmake libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy metadata first
COPY pyproject.toml README.md ./

RUN pip install --upgrade pip setuptools wheel
RUN pip install .

# Copy project files
COPY . .

CMD ["python", "-m", "pios_ml_backend.main", "run"]
