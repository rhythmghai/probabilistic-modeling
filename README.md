
---

## 🧠 Project Overview

This project models the **NO₂ air quality feature** using a **roll-number-parameterized non-linear transformation** and learns a **Gaussian-form probability density function (PDF)** from the transformed data.

The notebook demonstrates:
- Feature engineering with mathematical transformations
- Parametric density estimation using **Maximum Likelihood Estimation (MLE)**
- **Machine Learning-based validation** using gradient descent (PyTorch)
- Robust modeling using **clipping (outlier control)**
- Distribution diagnostics using **skewness, kurtosis, and visualization**

---

## 📌 Problem Statement

Given NO₂ values \(x\), apply the transformation:

\[
z = x + a_r \sin(b_r x)
\]

Where:

\[
a_r = 0.05 \cdot (r \bmod 7), \quad b_r = 0.3 \cdot (r \bmod 5 + 1)
\]

Then learn parameters of the Gaussian-form PDF:

\[
\hat{p}(z) = c \cdot e^{-\lambda (z - \mu)^2}
\]

Using both:
- **Statistical Estimation (MLE)**
- **Machine Learning Optimization (Gradient Descent)**

---

## ⚙️ Methodology

### 1. Data Preprocessing
- Remove missing NO₂ values
- Normalize using `StandardScaler` for numerical stability

### 2. Feature Transformation
- Apply a sine-based roll-number-parameterized transformation to introduce controlled non-linearity

### 3. Statistical PDF Learning (MLE)
- Compute mean and variance analytically
- Derive \(\lambda\) and \(c\) from Gaussian distribution theory

### 4. Robust Estimation
- Clip extreme values to reduce variance sensitivity to outliers

### 5. ML-Based Learning
- Define Gaussian negative log-likelihood
- Optimize \(\mu\) and \(\lambda\) using PyTorch’s Adam optimizer

### 6. Validation
- Compare MLE and ML parameters
- Visualize histogram vs learned PDFs
- Analyze skewness and kurtosis

---

## ▶️ How to Run

### Clone Repository
```bash
git clone https://github.com/your-username/no2-pdf-learning-roll-transform.git
cd no2-pdf-learning-roll-transform
