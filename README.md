# 🏠 Predict House Prices with Linear Regression

An end-to-end Machine Learning project demonstrating Multiple Linear Regression to predict median house prices in California districts using the **California Housing Dataset** and **Scikit-Learn**.

---

## 📋 Overview

This project implements a Multiple Linear Regression model to analyze and predict median house values (`MedHouseVal`) across various California block groups. By analyzing demographic, structural, and geographic attributes—such as median income, house age, average room count, and geographical coordinates—the model learns the mathematical relationship between housing features and market valuations.

---

## 📊 Dataset: California Housing

The dataset is loaded directly via `sklearn.datasets.fetch_california_housing`. It contains **20,640 records** with **8 input features** and **1 continuous target variable**.

### Features

| Feature | Description |
| :--- | :--- |
| **`MedInc`** | Median income in block group (in tens of thousands of USD) |
| **`HouseAge`** | Median house age in block group (years) |
| **`AveRooms`** | Average number of rooms per household |
| **`AveBedrms`** | Average number of bedrooms per household |
| **`Population`** | Block group population |
| **`AveOccup`** | Average number of household members |
| **`Latitude`** | Block group latitude coordinate |
| **`Longitude`** | Block group longitude coordinate |

### Target Variable

* **`MedHouseVal`**: Median house value for California districts, expressed in hundreds of thousands of dollars (\$100,000s).

---

## 🧠 Methodology: Multiple Linear Regression

Multiple Linear Regression models the linear relationship between several independent predictor variables ($X_1, X_2, \dots, X_n$) and the dependent target variable ($y$):

$$y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n + \epsilon$$

Where:
* $y$ is the predicted median house value (`MedHouseVal`)
* $\beta_0$ is the intercept term
* $\beta_1, \dots, \beta_n$ are the feature coefficients (weights)
* $X_1, \dots, X_n$ are the input housing features
* $\epsilon$ represents the residual error

The model minimizes the **Mean Squared Error (MSE)** via Ordinary Least Squares (OLS) to determine the best-fitting hyperplane.

---

## 📈 Project Pipeline

The workflow in [`program.ipynb`](./program.ipynb) follows these key steps:

1. **Data Loading & Inspection**: Fetching the dataset with `fetch_california_housing(as_frame=True)` and inspecting descriptive statistics using `df.describe()`.
2. **Feature & Target Separation**: Isolating independent feature matrix $X$ (all 8 features) and dependent target vector $y$ (`MedHouseVal`).
3. **Train-Test Split**: Splitting the 20,640 records into training (80%, 16,512 samples) and testing (20%, 4,128 samples) sets using `train_test_split(..., test_size=0.2, random_state=42)`.
4. **Model Training**: Fitting `LinearRegression()` on `X_train` and `y_train`.
5. **Model Evaluation**: Making predictions on `X_test` and evaluating performance using Mean Squared Error (MSE) and $R^2$ Score.
6. **Coefficient Interpretation**: Analyzing the model's intercept and learned feature coefficients.
7. **Inference on New Data**: Demonstrating single-sample prediction with custom feature inputs in a Pandas DataFrame.

---

## 📊 Performance Results

Evaluated on the test split (20% unseen data):

| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Mean Squared Error (MSE)** | `0.5559` | Average squared difference between actual and predicted valuations |
| **Coefficient of Determination ($R^2$)** | `0.5758` | The model explains **~57.58%** of the variance in California median house values |

### Model Parameters

* **Intercept ($\beta_0$)**: `-37.0233`

#### Feature Coefficients

| Feature | Coefficient | Impact & Direction |
| :--- | :--- | :--- |
| **`AveBedrms`** | `+0.7831` | Positive impact; higher average bedrooms relative to rooms correlates with higher value |
| **`MedInc`** | `+0.4487` | Strong positive impact; higher median income significantly increases home value |
| **`HouseAge`** | `+0.0097` | Slight positive impact on price |
| **`Population`** | `-0.000002` | Negligible linear impact |
| **`AveOccup`** | `-0.0035` | Slight negative correlation with occupancy count |
| **`AveRooms`** | `-0.1233` | Negative coefficient when controlling for bedroom count |
| **`Latitude`** | `-0.4198` | Geographic variation (southern locations generally trend higher) |
| **`Longitude`** | `-0.4337` | Geographic variation (coastal/western locations generally trend higher) |

---

## 🛠️ Tech Stack

* **Language**: Python 3.x
* **Machine Learning**: `scikit-learn`
* **Data Manipulation**: `pandas`, `numpy`
* **Development Environment**: Jupyter Notebook (`.ipynb`)

---

## 📦 Getting Started

### 1. Prerequisites & Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/<your-username>/Predict-House-Prices-with-Linear-Regression.git
cd Predict-House-Prices-with-Linear-Regression
pip install pandas numpy scikit-learn notebook
```

### 2. Run the Notebook

Launch Jupyter Notebook to view and run the analysis:

```bash
jupyter notebook program.ipynb
```

### 3. Quick Code Example

```python
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 1. Load California Housing Dataset
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# 2. Separate Features and Target
X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

# 3. Train-Test Split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make Prediction for New Input
new_data = pd.DataFrame({
    'MedInc': [5.0],
    'HouseAge': [40.0],
    'AveRooms': [5.0],
    'AveBedrms': [3.0],
    'Population': [5000.0],
    'AveOccup': [2.0],
    'Latitude': [37.88],
    'Longitude': [-122.23]
})

predicted_price = model.predict(new_data)
print(f"Predicted Median House Value: ${predicted_price[0] * 100000:,.2f}")
# Output: ~$443,508.18 (4.435 in $100k units)
```

---

## 📄 License

This project is licensed under the terms described in the [LICENSE](LICENSE) file.
