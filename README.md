# 🫁 Pneumonia Detection from Pediatric Chest X-Rays

> An AI-powered system for detecting pneumonia in children from chest X-rays, using multiple machine learning models and presented via an interactive Streamlit dashboard.

---

## 📸 Application Preview

(./ss.png)

---

## 🚀 Project Overview

This project provides a robust solution for **pneumonia detection in pediatric patients** through chest X-ray analysis. It combines **traditional ML classifiers** with a **deep CNN**, ensuring high accuracy.

**Workflow:**

1. Upload a pediatric chest X-ray.
2. The system runs six AI models (**CNN, Logistic Regression, SVM, Random Forest, KNN, Naive Bayes**).
3. Generates an **overall consensus prediction** and highlights the **most confident model**.
4. Shows **detailed model-wise predictions** with confidence scores in an interactive dashboard.

**Dataset:** 5,863 labeled X-rays from the Guangzhou Women and Children’s Medical Center, reviewed by multiple medical professionals to ensure quality.

This project is aimed at **providing affordable, accessible, and accurate screening for pediatric pneumonia**.

---

## ✨ Key Features

* **Multi-Model Analysis:** Six AI models run simultaneously for every X-ray.
* **Consensus Verdict:** Quick “Overall Prediction” and the **Most Confident Model** highlight.
* **Detailed Breakdown:** Interactive table showing model predictions and confidence scores.
* **Modern UI:** Custom dark theme with **glassmorphism-inspired cards** for a sleek experience.
* **Interactive Visuals:** Styled dataframes and progress bars for clear results interpretation.

---

## 🛠️ Tech Stack

* **Backend & Data Processing:** Python, Pandas, NumPy, OpenCV
* **Machine Learning:** Scikit-learn (KNN, Naive Bayes, Logistic Regression, SVM, Random Forest), TensorFlow/Keras (CNN)
* **Frontend & Dashboard:** Streamlit
* **Visualization:** Matplotlib, Seaborn

---

## ⚡ Installation & Setup

**1. Clone the repository:**

```bash
git clone https://github.com/your-username/Pneumonia-Detection-from-Chest-X-rays.git
cd Pneumonia-Detection-from-Chest-X-rays
```

**2. Set up a virtual environment:**

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4. Prepare the dataset:**

* Download from [Kaggle Chest X-Ray Pneumonia Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
* Place inside `data/` with structure:

```
data/chest_xray/train
data/chest_xray/test
data/chest_xray/val
```

---

## 🏃‍♂️ How to Run

**Option 1: Retrain Models (Optional)**

```bash
jupyter notebook "01-EDA-and-Preprocessing.ipynb"
```

*This regenerates model files in `models/`.*

**Option 2: Launch Streamlit Dashboard**

```bash
streamlit run app/app.py
```

* Browser opens automatically.
* Upload an X-ray to get **real-time predictions**.

---

## 📊 Model Performance

| Model               | Accuracy | Precision | Recall | F1-Score |
| ------------------- | :------: | :-------: | :----: | :------: |
| **CNN**             |  96.25%  |   95.92%  | 98.46% |  97.17%  |
| Logistic Regression |  95.00%  |   94.90%  | 97.44% |  96.15%  |
| SVM                 |  93.12%  |   92.02%  | 97.44% |  94.65%  |
| Random Forest       |  92.50%  |   91.18%  | 97.44% |  94.20%  |
| KNN (k=5)           |  88.75%  |   85.23%  | 93.75% |  89.29%  |
| Naive Bayes         |  86.88%  |   84.52%  | 94.87% |  89.40%  |

> The CNN performs the best, demonstrating the power of deep learning for medical image analysis.

---

## 📄 License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## ✨ Credits & Contact

* **Dataset:** Guangzhou Women and Children’s Medical Center
* **ML Libraries:** Scikit-learn, TensorFlow/Keras
* **Dashboard Framework:** Streamlit
* **Visualizations:** Matplotlib, Seaborn
* **Developer:** Abdulrehman Gulfaraz

**Contact Me:**

* GitHub: [https://github.com/abdulrehmangulfaraz](https://github.com/abdulrehmangulfaraz)
* LinkedIn: [https://www.linkedin.com/in/abdulrehman-gulfaraz](https://www.linkedin.com/in/abdulrehman-gulfaraz)

