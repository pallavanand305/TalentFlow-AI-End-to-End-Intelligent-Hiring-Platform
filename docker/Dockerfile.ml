FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy ML code
COPY ml/ ./ml/
COPY backend/app/models/ ./backend/app/models/
COPY backend/app/core/ ./backend/app/core/

# Run training script
CMD ["python", "-m", "ml.training.train_baseline"]
