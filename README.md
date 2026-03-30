# WSI-Segmentation-UNI2 🔬  
**Computational Pathology Pipeline for Prostate Cancer Reconstruction & Feature Extraction**

This repository provides a modular, high-throughput pipeline for processing Whole Slide Images (WSIs) using the **UNI2-h** foundation model. It handles the full lifecycle of digital pathology data: from raw slide patching to final diagnostic heatmap reconstruction.

---

## 🚀 Quick Start (Research & HPC Workflow)

### Installation & Environment Setup

Clone the repository and install dependencies (including `timm`, `huggingface_hub`, `h5py`, and `openslide`):

```bash
git clone https://github.com/yamlakyam/WSI-Segmentation-UNI2.git
cd WSI-Segmentation-UNI2
bash setup.sh