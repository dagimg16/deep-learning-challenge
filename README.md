# 📊 Alphabet Soup Charity: Deep Learning Classifier

## 🧠 Overview

Alphabet Soup, a nonprofit foundation, wants to improve how it selects funding applicants by identifying those most likely to succeed. Using historical data from over 34,000 organizations, this project builds a binary classification model using TensorFlow and Keras to predict the success of applicants based on features like application type, organization affiliation, income level, and more.

---

## 📁 Dataset

The dataset includes over 34,000 records and the following key features:

- `APPLICATION_TYPE`, `AFFILIATION`, `CLASSIFICATION`, `USE_CASE`, `ORGANIZATION`, etc.
- `ASK_AMT`: Funding requested
- `INCOME_AMT`: Income range of applicant
- `IS_SUCCESSFUL`: **Target variable** — whether the funding was used successfully (1 = Yes, 0 = No)

---

## ⚙️ Preprocessing Steps

- Dropped non-predictive columns: `EIN`, `NAME`
- Grouped rare categories into `"Other"` for:
  - `CLASSIFICATION` (values with < 1,800 instances)
  - `APPLICATION_TYPE` (values with < 500 instances)
- Encoded categorical variables using one-hot encoding (`pd.get_dummies`)
- Scaled features using `StandardScaler`
- Split data into training and testing sets (75% / 25%)

---

## 🤖 Model Architecture

A deep neural network was built using Keras `Sequential` API:

| Layer Type         | Units | Activation | Notes                          |
|--------------------|-------|------------|--------------------------------|
| Dense              | 128   | `tanh`     | Input layer with normalization |
| BatchNormalization | -     | -          | Normalizes input features      |
| Dropout            | -     | -          | Dropout rate: 0.2              |
| Dense              | 64    | `tanh`     | Hidden layer                   |
| BatchNormalization | -     | -          |                                |
| Dropout            | -     | -          | Dropout rate: 0.2              |
| Dense              | 32    | `tanh`     | Hidden layer                   |
| Dense              | 1     | `sigmoid`  | Output layer (binary)          |

**Additional Details:**
- Optimizer: `RMSprop`
- Loss function: `binary_crossentropy`
- Regularization: `Dropout` + `BatchNormalization`
- EarlyStopping: Enabled (patience = 10)
- Epochs: 200 max 
- Batch size: 64

---

## 📊 Performance

| Metric           | Score     |
|------------------|-----------|
| Final Accuracy   | **72.89%** |
| Final Loss       | ~0.558    |
| Target Accuracy  | 75%       |

Although the model did not reach 75%, it demonstrated strong performance with careful tuning and regularization.

---

## 🔁 Optimization Techniques Used

- Increased model depth and neuron count
- Tested multiple activation functions (`relu`, `tanh`, `LeakyReLU`)
- Applied `Dropout` and `BatchNormalization`
- Switched optimizers (`Adam`, `RMSprop`, `Adamax`)
- Early stopping based on validation loss
- Grouped rare categories into `"Other"` to reduce noise

---

## 💡 Future Recommendations

For improved performance, consider switching to:
- **XGBoost** or **Random Forest** for tabular data
- **Feature importance analysis** to prune noisy inputs

These alternatives often perform better on structured datasets and provide better model interpretability for stakeholders.

---

## 💾 Output

The final optimized model is saved as: **AlphabetSoupCharity_Optimization.h5**

