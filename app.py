import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder

# Set up the Streamlit app layout
st.title('Customer Churn Prediction Dashboard')

# Upload CSV file
uploaded_file = st.file_uploader("Upload Customer Data CSV", type=["csv"])

if uploaded_file is not None:
    try:
        # Load the data into a pandas DataFrame
        df = pd.read_csv(uploaded_file)

        

        st.subheader("📊 Dataset Preview")
        st.write(df.head())

        # EDA Section
        st.subheader("🔍 Exploratory Data Analysis (EDA)")
        
        st.write("**Shape of Dataset:**", df.shape)
        st.write("**Columns:**", df.columns.tolist())
        
        st.write("**Missing Values:**")
        st.write(df.isnull().sum())
        
        st.write("**Churn Distribution:**")
        st.bar_chart(df['Churn'].value_counts())

        st.write("**Numerical Feature Summary:**")
        st.write(df.describe())


        # Encode categorical variables
        label_encoder = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = label_encoder.fit_transform(df[col])

        # Features and target
        X = df.drop('Churn', axis=1)
        y = df['Churn']

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)

        st.subheader("✅ Model Evaluation")
        st.write(f"**Model Accuracy:** {accuracy * 100:.2f}%")

        # Confusion Matrix
        st.write("**Confusion Matrix:**")
        cm = confusion_matrix(y_test, y_pred)
        fig2, ax2 = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig2)

        # Classification Report
        st.write("**Classification Report:**")
        st.text(classification_report(y_test, y_pred))

        # ROC Curve
        st.write("**ROC Curve:**")
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig3, ax3 = plt.subplots()
        ax3.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc_score(y_test, y_proba):.2f})")
        ax3.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        st.pyplot(fig3)

        # Sidebar - Input new customer details
        st.sidebar.title("📥 Enter New Customer Details")
        
        gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
        senior_citizen = st.sidebar.selectbox('Senior Citizen (0 or 1)', ['0', '1'])
        partner = st.sidebar.selectbox('Partner', ['Yes', 'No'])
        dependents = st.sidebar.selectbox('Dependents', ['Yes', 'No'])
        tenure = st.sidebar.slider('Tenure (Months)', 1, 100)
        phone_service = st.sidebar.selectbox('Phone Service', ['No', 'Yes'])
        multiple_lines = st.sidebar.selectbox('Multiple Lines', ['No phone service', 'Yes', 'No'])
        internet_service = st.sidebar.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
        online_security = st.sidebar.selectbox('Online Security', ['Yes', 'No', 'No internet'])
        online_backup = st.sidebar.selectbox('Online Backup', ['Yes', 'No', 'No internet'])
        device_protection = st.sidebar.selectbox('Device Protection', ['Yes', 'No', 'No internet'])
        tech_support = st.sidebar.selectbox('Tech Support', ['Yes', 'No', 'No internet'])
        streaming_tv = st.sidebar.selectbox('Streaming TV', ['Yes', 'No', 'No internet'])
        streaming_movies = st.sidebar.selectbox('Streaming Movies', ['Yes', 'No', 'No internet'])
        contract = st.sidebar.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
        paperless_billing = st.sidebar.selectbox('Paperless Billing', ['Yes', 'No'])
        payment_method = st.sidebar.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
        monthly_charges = st.sidebar.number_input('Monthly Charges', min_value=1.0, max_value=200.0, value=50.0)
        total_charges = st.sidebar.number_input('Total Charges', min_value=1.0, max_value=10000.0, value=500.0)

        input_data = pd.DataFrame({
            'gender': [gender],
            'SeniorCitizen': [senior_citizen],
            'Partner': [partner],
            'Dependents': [dependents],
            'tenure': [tenure],
            'PhoneService': [phone_service],
            'MultipleLines': [multiple_lines],
            'InternetService': [internet_service],
            'OnlineSecurity': [online_security],
            'OnlineBackup': [online_backup],
            'DeviceProtection': [device_protection],
            'TechSupport': [tech_support],
            'StreamingTV': [streaming_tv],
            'StreamingMovies': [streaming_movies],
            'Contract': [contract],
            'PaperlessBilling': [paperless_billing],
            'PaymentMethod': [payment_method],
            'MonthlyCharges': [monthly_charges],
            'TotalCharges': [total_charges]
        })

        for col in input_data.columns:
            if input_data[col].dtype == 'object':
                input_data[col] = label_encoder.fit_transform(input_data[col])

        input_data = input_data[X.columns]

        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.sidebar.write("🚨 **Prediction: Customer is likely to churn!**")
        else:
            st.sidebar.write("✅ **Prediction: Customer is likely to stay.**")

        # Business Insights Section
        st.subheader("💡 Business Insights")
        st.write("""        
        - Customers with **month-to-month contracts** and **electronic checks** show higher churn rates.
        - **Senior citizens** and customers with **low tenure** are more likely to churn.
        - **Technical support**, **online security**, and **device protection** help in reducing churn.
        - High **monthly charges** combined with **low total charges** (new customers) tend to churn more easily.
        
        📢 **Actionable Recommendation:**  
        Focus retention offers on new customers, especially those with month-to-month contracts, 
        and promote bundled services like online security and technical support to reduce churn.
        """)

    except Exception as e:
        st.error(f"Error: {e}")
