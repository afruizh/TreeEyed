# Dockerfile for TreeEyed
#FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app


# Install system dependencies for PySide6 (Qt for Python) and Miniconda prerequisites
RUN apt-get update && apt-get install -y wget bzip2 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# # Install Miniconda
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
# 	bash ~/miniconda.sh -b -p /opt/conda && \
# 	rm ~/miniconda.sh
# ENV PATH=/opt/conda/bin:$PATH

# Install Miniforge
RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O ~/miniforge.sh && \
    bash ~/miniforge.sh -b -p /opt/conda && \
    rm ~/miniforge.sh
ENV PATH=/opt/conda/bin:$PATH

# Copy environment.yml and create conda environment
COPY environment.yml ./
RUN conda env create -f environment.yml

# Copy TreeEyed source code
COPY src/ ./src/

CMD ["bash", "--init-file", "/opt/conda/etc/profile.d/conda.sh"]
