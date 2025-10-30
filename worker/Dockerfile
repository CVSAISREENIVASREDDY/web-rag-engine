# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt with increased timeout
RUN pip install --default-timeout=1000 --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the rest of the application's code into the container at /app
COPY ./app /app/app

# Command to run the Celery worker
CMD ["celery", "-A", "app.celery_app.celery_app", "worker", "--loglevel=info"]