import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Fraud Detection", layout="centered")
st.title("💳 Fraud Detection")

st.write("🔍 This app uses a logistic regression model.")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard_small.csv").dropna()
    return df

df = load_data()

# Show preview
with st.expander("📄 View Data Preview"):
    st.dataframe(df.head())

# Split into features and labels
X = df.drop(columns="Class")
y = df["Class"]

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Evaluate
train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

with st.expander("📊 Model Performance"):
    st.write(f"*Training Accuracy:* {train_acc:.4f}")
    st.write(f"*Testing Accuracy:* {test_acc:.4f}")

# Upload new data for prediction
st.subheader("📁 Upload New Transactions to Predict")
uploaded_file = st.file_uploader("Upload a CSV (same columns as input features, no 'Class')", type=["csv"])

if uploaded_file:
    input_data = pd.read_csv(uploaded_file)
    predictions = model.predict(input_data)
    input_data["Prediction"] = ["Fraud" if p == 1 else "Not Fraud" for p in predictions]

    st.success("🎯 Prediction completed.")
    st.dataframe(input_data)

    csv = input_data.to_csv(index=False).encode()
    st.download_button("📥 Download Results", csv, "fraud_predictions.csv", "text/csv")

