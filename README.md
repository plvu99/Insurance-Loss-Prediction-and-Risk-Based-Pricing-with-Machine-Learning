# Insurance Loss Prediction and Risk-Based Pricing with Machine Learning

## 🔎 Overview

Insurance companies must accurately estimate expected claim losses to price policies fairly and remain profitable in a competitive market. This project develops **machine learning** models to **predict insurance losses and claim likelihood** using a large policyholder dataset.

The analysis focuses on predicting three key insurance metrics:

* **Loss Cost (LC)** – expected claim loss per exposure unit
* **Historically Adjusted Loss Cost (HALC)** – loss cost adjusted by claim frequency
* **Claim Status (CS)** – whether a policyholder files a claim

By applying multiple machine learning models and comparing their performance, the project demonstrates how data-driven approaches can improve underwriting accuracy and risk-based pricing strategies.

## 🔐 Business Problem

One of the biggest challenges in insurance is setting premiums that are both fair and profitable. If insurers misprice risk:

* Low-risk customers may leave for competitors
* High-risk customers remain
* Insurers face adverse selection and financial losses

Accurately predicting expected claim losses allows insurers to:

* price policies according to risk
* segment customers more effectively
* maintain a balanced risk portfolio

However, insurance data presents several challenges:

* Highly skewed claim distributions
* Many zero claims
* Complex nonlinear relationships between risk factors

This project addresses these challenges by developing predictive models tailored to insurance loss data. 

## 📊 Dataset

The dataset contains insurance policy transactions with demographic, policy, and vehicle information.
- Training dataset: 37,451 policy records, 28 predictor variables
- Test dataset: 15,787 policy records

Key variables include:

**Policyholder characteristics**
* Policy tenure
* Number of active policies
* Claim history

**Vehicle information**
* Vehicle type
* Vehicle value
* Vehicle power
* Cylinder capacity

**Insurance behavior**
* Claim counts
* Claim costs
* Payment methods

The dataset also includes variables needed to compute the main targets:

**Loss Cost (LC)**

```
LC = Total Claim Cost / Number of Claims
```

**Historically Adjusted Loss Cost (HALC)**

```
HALC = LC × Claim Frequency
```

These metrics help insurers evaluate both severity and frequency of losses. 

## 📍 Methodology

The modeling pipeline consisted of four main stages.

### 1. Data Preprocessing

Several feature engineering steps were applied:
* Converted date variables to datetime format
* Created derived features:
  * policyholder age
  * driver experience
  * policy duration
  * vehicle age
* Encoded categorical variables
* One-hot encoded features for modeling

Missing values in fuel type were imputed using XGBoost classification.

Highly correlated variables were removed to reduce multicollinearity.

Outliers were handled by binning extreme values to stabilize model training. 

### 2. Exploratory Data Analysis

EDA revealed important patterns in the insurance data:

* Claim cost variables show strong right skew.
* Most policyholders do not file claims (most claims are zero).
* Only a small subset generate very large losses.
* Vehicle and policy features are strong predictors of risk.

<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/d6168c28-8758-4d5c-986c-7b5b379cce23" />

These characteristics confirm the need for models capable of handling zero-inflated and skewed distributions. 

### 3. Regression Models

Regression models were developed to predict:

* **Loss Cost (LC)**
* **Historically Adjusted Loss Cost (HALC)**

Models tested:

* Tweedie Regression
* LightGBM
* XGBoost
* Neural Networks

All models were trained using:

* Tweedie loss function
* Feature scaling
* 5-fold cross-validation

A two-stage modeling approach was used:

1. Predict LC
2. Use predicted LC plus original features to estimate HALC

### 4. Classification Models

Classification models predicted **Claim Status (CS)**.

Models tested:

* Logistic Regression
* Random Forest
* XGBoost
* Neural Networks

Model performance was evaluated using:

* ROC-AUC
* Cross-validation

## 🔑 Key Insights

### Regression Results

LightGBM delivered the best performance:

| Model              | LC MSE     | HALC MSE    |
| ------------------ | ---------- | ----------- |
| Tweedie Regression | 502.30     | 1020.02     |
| **LightGBM**       | **500.82** | **1019.12** |
| XGBoost            | 503.78     | 1025.16     |
| Neural Networks    | 502.45     | 1019.48     |

<img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/5735be05-be7b-49a7-af9f-6922530861a6" />

<img width="788" height="590" alt="image" src="https://github.com/user-attachments/assets/6afef0d2-b678-4718-9362-bdfe59b70dcb" />

Tree-based ensemble models capture nonlinear patterns better than traditional statistical models.

### Classification Results

XGBoost achieved the best claim prediction performance:

| Model               | ROC-AUC  |
| ------------------- | -------- |
| Logistic Regression | 0.71     |
| **XGBoost**         | **0.74** |
| Random Forest       | 0.73     |
| Neural Networks     | 0.73     |

<img width="790" height="590" alt="image" src="https://github.com/user-attachments/assets/a66b788a-2d4c-4867-90c1-acc041c7f95b" />

This indicates strong separation between claim vs non-claim policyholders.

Here is the [LC, HALC, CS prediction](https://github.com/plvu99/Car-Insurance-Loss-Analytics/blob/main/insurance_prediction.csv).

## ✍️ Business Recommendations

### 1. Implement risk-based pricing

Using predicted LC and HALC allows insurers to price policies more accurately according to expected losses.

### 2. Improve underwriting decisions

Predicting claim likelihood enables insurers to identify higher-risk policyholders and adjust underwriting policies accordingly.

### 3. Enhance customer segmentation

Machine learning models allow insurers to group policyholders by risk profiles, enabling personalized pricing and better portfolio management.

### 4. Deploy real-time pricing models

Ensemble models such as LightGBM and XGBoost can be integrated into pricing engines to support real-time underwriting decisions.

## ⚙ Tools & Techniques

* Python
* Data preprocessing (Pandas, NumPy)
* Data visualization (Matplotlib, Seaborn)
* Machine learning (LightGBM, XGBoost, Tweedie Regression, Neural Networks, Random Forest, Logistic Regression)
* Model evaluation (Mean Squared Error, ROC-AUC, Cross-Validation)
