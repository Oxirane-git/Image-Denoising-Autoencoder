# Image-Denoising-Autoencoder
This project implements a deep learning pipeline in PyTorch for denoising grayscale document images (like scanned IDs, licenses, etc.) using an autoencoder architecture inspired by U-Net and enhanced with residual blocks. The goal is to remove noise and restore clean document images for improved readability and OCR performance.
# 🧼 Image Denoising using Autoencoder with U-Net Architecture

This project is a PyTorch-based implementation of an image denoising system for **grayscale document images** (such as scanned Aadhar cards, PAN cards, driving licenses, etc.). It uses an **autoencoder** with **U-Net-style skip connections** and **residual blocks** for high-quality denoising, especially in real-world noisy document restoration tasks.

---

## 🔍 Project Overview

📌 **Objective**:  
To clean noisy grayscale document images and restore them to a clearer, more readable format for better visual clarity and OCR performance.

📌 **Highlights**:
- Custom **Encoder-Decoder architecture** with skip connections
- **Residual blocks** for deep bottleneck feature refinement
- Combined **MSE + L1 + Gradient loss** for sharper denoising
- End-to-end training pipeline in PyTorch
- Ready-to-use for scanned documents or degraded images

---

## 🧠 Model Architecture

- **Encoder**:
  - Sequential convolutional layers with ReLU and BatchNorm
  - MaxPooling for downsampling
- **Bottleneck**:
  - Residual blocks with identity connections
- **Decoder**:
  - Transposed convolutions for upsampling
  - Skip connections (U-Net style) from encoder layers
- **Final Output**:
  - Single-channel grayscale image output with sigmoid activation

---

## 📁 Folder Structure

