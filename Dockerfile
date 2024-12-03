# Start from an official Python image with Jupyter capabilities
FROM python:3.12-slim

# Set environment variables to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install necessary dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Install Python packages
RUN pip install \
    numpy \
    pandas \
    seaborn \
    matplotlib \
    ipython \
    scikit-learn \
    scipy \
    jupyter

# Set the working directory
WORKDIR /workspace

# Copy your project files into the container
COPY . /workspace

# Set environment variable for matplotlib backend (to prevent display errors in non-GUI environments)
ENV MPLBACKEND=Agg

# Optional: Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Optional: Command to start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]