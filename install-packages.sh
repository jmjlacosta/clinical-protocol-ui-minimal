#!/bin/bash
set -e

# Update package list
apt-get update

# Install system dependencies for PDF processing and other requirements
apt-get install -y \
    build-essential \
    libpoppler-cpp-dev \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    libmagic1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh