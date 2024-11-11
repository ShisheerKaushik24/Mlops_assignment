# Use an official Python image from Docker Hub
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the local files to the container
COPY . /app

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create a folder to store the output figures
RUN mkdir -p /app/figures

# Run the Python script
CMD ["python", "model.py"]