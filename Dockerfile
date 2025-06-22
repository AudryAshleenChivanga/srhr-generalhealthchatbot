# Use official Python base image with desired version
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the entire app code into container
COPY . .

# Expose the port Streamlit listens on
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "ashlebot.py", "--server.port=8501", "--server.enableCORS=false"]
