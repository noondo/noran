# Use an official lightweight Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the entire project into the container
COPY . /app

# Install the 'uv' package manager
RUN pip install uv

# Install project dependencies using uv
RUN uv sync

# Expose port 8000 for the FastAPI server
EXPOSE 8000

# Command to start the FastAPI application
CMD ["uv", "run", "uvicorn", "production_app.api:app", "--host", "0.0.0.0", "--port", "8000"]