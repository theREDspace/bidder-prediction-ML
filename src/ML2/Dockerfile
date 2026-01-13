# Use a slim Python image to keep size down
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies required for LightGBM (libgomp1)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the python script
COPY predict_bids.py .

# Command to run the script when the container starts
CMD ["python", "predict_bids.py"]