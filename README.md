# House Price Prediction

An end-to-end Machine Learning project that predicts house prices using the Kaggle House Prices dataset.

## Project Overview

The goal of this project is to build a machine learning model that can predict the selling price of a house based on its features such as overall quality, living area, garage capacity, basement area, number of bathrooms, year built, and neighborhood.

The project covers the complete machine learning workflow from data exploration and preprocessing to model training, evaluation, hyperparameter tuning, model saving, and deployment using Streamlit.

## Dataset

Dataset: Kaggle House Prices - Advanced Regression Techniques

The dataset contains information about residential properties in Ames, Iowa.

## Machine Learning Workflow

1. Data Loading
2. Exploratory Data Analysis
3. Missing Value Handling
4. Outlier Analysis
5. Feature Selection
6. Categorical Feature Encoding
7. Train-Test Split
8. Model Training
9. Model Comparison
10. Hyperparameter Tuning
11. Log Transformation of Target
12. Model Evaluation
13. Final Model Selection
14. Model Serialization
15. Streamlit Deployment

## Models Used

- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Gradient Boosting performed strongly on the held-out test set and was selected as the final model.

## Evaluation

The project uses RMSLE (Root Mean Squared Logarithmic Error), which is the evaluation metric used in the Kaggle House Prices competition.

The final tuned Gradient Boosting pipeline achieved an RMSLE of approximately **0.1171** on the held-out test set.

## Deployment

The trained model was saved using Joblib and deployed as an interactive Streamlit web application.

The application allows users to enter important house details and receive an estimated house price.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

## Project Structure

```text
HousePricePrediction/
│
├── app.py
├── house_price_prediction.ipynb
├── house_price_model.pkl
├── submission.csv
├── requirements.txt
└── README.md

## How to Run

### 1. Install the required dependencies

```bash
pip install -r requirements.txt

streamlit run app.py