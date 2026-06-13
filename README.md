# INF-74-51-M-6-Embedded-intelligence

# 🧠 Embedded Intelligence — RPTU Kaiserslautern

> *Teaching machines to understand the physical world — one sensor at a time.*

This repository contains coursework for the **Embedded Intelligence** module at RPTU Kaiserslautern, developed as part of the MSc Informatics programme. The exercises explore the full pipeline of building intelligent systems from raw sensor data — from signal preprocessing and feature extraction to model training and evaluation on real-world human activity datasets.

The work is grounded in a research dataset developed at **DFKI (German Research Center for Artificial Intelligence)** and covers techniques at the intersection of signal processing, machine learning, and embedded systems.

---

## 📂 Repository Structure

```
embedded-intelligence/
│
├── Exercise-1/          # Data Exploration & Sliding Window Preprocessing
├── Exercise-2/          # Feature Extraction
├── Exercise-3/          # Classical ML Classification
├── Exercise-4/          # Deep Learning for HAR
├── Exercise-5/          # Model Evaluation & Optimisation
├── Exercise-6/          # Deployment & Embedded Inference
│
└── README.md            # You are here
```

---

## 📋 Exercises at a Glance

| # | Topic | Key Concepts | Dataset |
|---|-------|-------------|---------|
| [01](./Exercise-1/) | Data Exploration & Windowing | Sliding windows, NaN handling, IMU signals | PAMAP2 |
| [02](./Exercise-2/) | Feature Extraction | Time-domain & frequency-domain features | PAMAP2 |
| [03](./Exercise-3/) | Classical ML Classification | SVM, Random Forest, kNN | PAMAP2 |
| [04](./Exercise-4/) | Deep Learning for HAR | CNN, LSTM, sequence modelling | PAMAP2 |
| [05](./Exercise-5/) | Evaluation & Optimisation | Confusion matrices, hyperparameter tuning | PAMAP2 |
| [06](./Exercise-6/) | Embedded Deployment | Model compression, on-device inference | PAMAP2 |

---

## 🗃️ Dataset

All exercises use the **PAMAP2 Physical Activity Monitoring Dataset**, originally published by researchers at DFKI and hosted on the UCI Machine Learning Repository.

- **Subjects:** 9 participants
- **Sensors:** IMU units on hand, chest, and ankle (100Hz)
- **Activities:** 18 physical activities including walking, running, cycling, and household tasks
- **Source:** [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/PAMAP2+Physical+Activity+Monitoring)

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=flat-square)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)

---

## ▶️ How to Run

Each exercise is a self-contained Jupyter notebook. To run:

1. Clone this repository
2. Upload the PAMAP2 dataset to your Kaggle account or Google Drive
3. Update the `DATASET_PATH` variable in the notebook to point to your dataset location
4. Run all cells in order

```bash
git clone https://github.com/arnavgithub1911/INF-74-51-M-6-Embedded-Intelligence
cd INF-74-51-M-6-Embedded-Intelligence/Exercise-1
jupyter notebook
```

---

*Course: Embedded Intelligence · RPTU Kaiserslautern · MSc Informatics*
*Instructor dataset credit: Bo Zhou, DFKI*
