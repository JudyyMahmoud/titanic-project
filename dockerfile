# 1. Start from a lightweight Python 3.9 image
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy requirements first (for better caching)
COPY requirments.txt .

# 4. Install dependencies
RUN pip install --no-cache-dir -r requirments.txt

# 5. Copy the source code and the trained model
COPY src/ ./src/
COPY model1.pkl .

# 6. Expose the port the app runs on
EXPOSE 8000

# 7. Command to run the app
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]