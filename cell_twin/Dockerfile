FROM python:3.10

WORKDIR /app

# Copy requirements
COPY backend/requirements.txt .
RUN pip install -r requirements.txt

# Copy backend code
COPY backend /app

# Expose port
EXPOSE 8000

# Run server
CMD ["python", "main.py"]