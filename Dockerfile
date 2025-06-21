# 1. Use an official Python 3.10 slim base image
FROM python:3.10-slim

# 2. Set a working directory inside the container
WORKDIR /app

# 3. Copy the requirements file to the container
COPY requirements.txt .

# 4. Install system dependencies needed for unzip and TensorFlow
RUN apt-get update && apt-get install -y \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 5. Upgrade pip and install Python dependencies from requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 6. Copy your app source code into the container
COPY . .

# 7. Expose port 8501 for Streamlit
EXPOSE 8501

# 8. Run the Streamlit app when the container launches
CMD ["streamlit", "run", "ashlebot.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
