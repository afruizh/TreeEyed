# Dockerfile for TreeEyed
#FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install Python and pip
RUN apt-get update && \
	apt-get install -y python3 python3-pip && \
	rm -rf /var/lib/apt/lists/*

# Symlink python to python3 for compatibility
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install system dependencies for PySide6 (Qt for Python)
RUN apt-get update && apt-get install -y libglib2.0-0

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy TreeEyed source code
COPY src/ ./src/

# Set environment variables if needed
# ENV PYTHONPATH=/app/src

# Default command to run tree_eyed_app.py
#CMD ["python", "src/tree_eyed/tree_eyed_app.py"]
