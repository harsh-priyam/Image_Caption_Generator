FROM python:3.11.4-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install wget and pip dependencies
RUN pip install --no-cache-dir gdown

# Set work directory
WORKDIR /app

# Create artifacts directory
RUN mkdir -p artifacts

# Download model from Google Drive
RUN gdown --id 1V10sk5xQnEzkSJLxJaDUaBkGxb1x9l72 -O artifacts/model.pth

# Copy project files
COPY . /app

# Install project dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port your app runs on
EXPOSE 8000

# Use Gunicorn as the WSGI server
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app"]