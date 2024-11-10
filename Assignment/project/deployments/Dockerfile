# Base image
FROM python:3.9-slim

# Working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Start Streamlit app
CMD ["streamlit", "run", "deployments/app.py"]
