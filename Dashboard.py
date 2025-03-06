import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from lifelines import KaplanMeierFitter
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set up Streamlit page
st.set_page_config(page_title="AI-Powered Cancer Diagnosis Dashboard", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select a Page", ["📊 KPI Dashboard", "📈 Data Exploration", "🚀 Predictive Modeling & Anomaly Detection"])

# Upload File
uploaded_file = st.file_uploader("Upload a dataset (CSV or Excel)", type=["csv", "xlsx"])

def load_data(uploaded_file):
    if uploaded_file is not None:
        file_extension = uploaded_file.name.split(".")[-1]  # Get file extension
        if file_extension == "csv":
            df = pd.read_csv(uploaded_file)
        elif file_extension == "xlsx":
            df = pd.read_excel(uploaded_file, sheet_name="Dataset")
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
    else:
        st.warning("No file uploaded. Loading default dataset...")
        df = pd.read_excel("Breast Cancer_Analysis_Dashboard.xlsx", sheet_name="Dataset")  # Ensure this file exists
    return df

df = load_data(uploaded_file)

# Show dataset preview
if df is not None:
    st.write("Dataset Preview:", df.head())
if page == "📊 KPI Dashboard":
    st.title("📊 KPI Dashboard - Cancer Diagnosis")
    if df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # KPI Analysis
        st.subheader("📈 Time Trends (Line Chart)")
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            time_trend = df.groupby(df["Date"].dt.to_period("M")).size()
            st.line_chart(time_trend)
        else:
            st.warning("Date column not found. Ensure dataset includes a time variable.")

        # Pie Chart - Demographic Breakdown
        st.subheader("📊 Demographic Breakdown")
        category = st.selectbox("Select a categorical column", df.select_dtypes(include=['object']).columns)
        category_counts = df[category].value_counts()
        fig, ax = plt.subplots()
        ax.pie(category_counts, labels=category_counts.index, autopct="%1.1f%%")
        st.pyplot(fig)

elif page == "📈 Data Exploration":
    st.title("📈 Data Exploration")
    if df is not None:
        # Descriptive Statistics
        st.subheader("📊 Descriptive Statistics")
        st.dataframe(df.describe())
        
        # Heatmap
        st.subheader("📈 Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("No numerical columns found for correlation analysis.")
        
        # Histogram
        st.subheader("📊 Histogram")
        hist_col = st.selectbox("Select a column for histogram", numeric_df.columns)
        fig, ax = plt.subplots()
        sns.histplot(df[hist_col], kde=True, bins=30, ax=ax)
        ax.set_title(f"Histogram of {hist_col}")
        st.pyplot(fig)

        # Scatter Plot
        st.subheader("📌 Scatter Plot")
        col1, col2 = st.columns(2)
        x_col = col1.selectbox("Select X-axis Variable", numeric_df.columns)
        y_col = col2.selectbox("Select Y-axis Variable", numeric_df.columns)
        fig, ax = plt.subplots()
        sns.scatterplot(x=df[x_col], y=df[y_col], alpha=0.6)
        ax.set_title(f"Scatter Plot: {x_col} vs {y_col}")
        st.pyplot(fig)

        # Box Plot
        st.subheader("📌 Box Plot")
        box_col = st.selectbox("Select a column for box plot", numeric_df.columns)
        fig, ax = plt.subplots()
        sns.boxplot(y=df[box_col], ax=ax)
        ax.set_title(f"Box Plot of {box_col}")
        st.pyplot(fig)

        # Kaplan-Meier Survival Analysis
        st.subheader("📉 Kaplan-Meier Survival Curve")
        if "Survival_Years" in df.columns and "Diagnosis" in df.columns:
            kmf = KaplanMeierFitter()
            diagnosed = df[df["Diagnosis"] == "Positive"]
            kmf.fit(diagnosed["Survival_Years"], event_observed=[1] * len(diagnosed))
            fig, ax = plt.subplots()
            kmf.plot_survival_function(ax=ax)
            ax.set_title("Kaplan-Meier Survival Curve")
            ax.set_xlabel("Years")
            ax.set_ylabel("Survival Probability")
            st.pyplot(fig)
        else:
            st.warning("Survival data not found. Ensure dataset includes Survival_Years and Diagnosis columns.")

        # Generate and Export Report
        st.subheader("📥 Download Data Analysis Report")
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
            df.describe().to_excel(writer, sheet_name="Descriptive Statistics")
            numeric_df.corr().to_excel(writer, sheet_name="Correlation Matrix")
        output.seek(0)
        st.download_button(
            label="Download Report (Excel)",
            data=output,
            file_name="Data_Analysis_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

elif page == "🚀 Predictive Modeling & Anomaly Detection":
    st.title("🚀 AI-Powered Prediction & Anomaly Detection")
    if df is not None:
        # Encode categorical variables
        df_encoded = df.copy()
        label_encoders = {}
        for col in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        
        # Train-Test Split
        if "Diagnosis" in df_encoded.columns:
            X = df_encoded.drop(columns=["Diagnosis"])
            y = df_encoded["Diagnosis"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train Random Forest Model
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            st.subheader("📊 Model Accuracy")
            st.write(f"Random Forest Model Accuracy: {accuracy * 100:.2f}%")
            
            # Anomaly Detection (Detect Outliers)
            st.subheader("⚠️ Anomaly Detection")
            anomalies = df_encoded[(df_encoded["Survival_Years"] < df_encoded["Survival_Years"].quantile(0.05))]
            st.write(f"Detected {len(anomalies)} potential anomalies in survival data.")
            st.dataframe(anomalies)
        else:
            st.warning("Diagnosis column missing. Ensure dataset includes a target variable for predictions.")

st.success("✅ AI-Powered Cancer Dashboard Ready!")




                                                                                   