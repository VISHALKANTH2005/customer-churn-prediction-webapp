#Customer Churn Prediction Web App

This is an interactive Streamlit web application that predicts whether a customer is likely to churn (leave) based on their service usage and contract details. The project uses machine learning (Random Forest) and offers visual insights through EDA (Exploratory Data Analysis).

---

Features

- Upload your customer dataset (CSV format)
- Automatic data preprocessing and label encoding
- Exploratory Data Analysis:
  - Shape and preview of dataset
  - Missing value detection
  - Churn distribution chart
  - Summary statistics of numerical columns
- Machine Learning Model:
  - Random Forest Classifier
  - Accuracy score, Confusion Matrix, ROC Curve, Classification Report
- Real-time churn prediction for new customers via sidebar input
- Actionable business insights based on model outputs

---

 Tech Stack

- Python
- Streamlit
- Pandas, NumPy
- scikit-learn (RandomForest)
- Matplotlib, Seaborn

---

 How to Run the App

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/customer-churn-prediction-webapp.git
   cd customer-churn-prediction-webapp
