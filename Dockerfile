FROM python:3.13-slim

WORKDIR /app

# Install system dependencies (if needed)
RUN apt-get update && apt-get install -y build-essential

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app
COPY . .

# Expose the port
EXPOSE 8080

# Start the app using uvicorn (installed globally)
CMD ["python3", "qdrant.py"]
