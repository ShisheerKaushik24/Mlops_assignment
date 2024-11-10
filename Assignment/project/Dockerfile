# Use official Python image from the Docker Hub
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy source code and data
COPY . .

# Install dependencies
RUN pip install -r requirements.txt

# Run the training script
CMD ["python", "src/train_model.py"]
