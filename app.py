import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Fraud Detection App", layout="centered")

st.title("🔍 Credit Card Fraud Detection")
st.write("This app uses logistic regression to detect fraudulent transactions.")

# Load and preprocess the dataset
@st.cache_data
def load_data():
    data = pd.read_csv('creditcard.csv')
    data = data.dropna()  # remove missing values
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]
    legit_sample = legit.sample(n=fraud.shape[0]*5)
    new_data = pd.concat([legit_sample, fraud], axis=0)
    X = new_data.drop(columns='Class')
    Y = new_data['Class']
    return train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

X_train, X_test, Y_train, Y_test = load_data()

# Train the model
@st.cache_resource
def train_model(X_train, Y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, Y_train)
    return model

model = train_model(X_train, Y_train)

# Accuracy Display
X_train_pred = model.predict(X_train)
X_test_pred = model.predict(X_test)

train_acc = accuracy_score(Y_train, X_train_pred)
test_acc = accuracy_score(Y_test, X_test_pred)

with st.expander("📊 Model Performance"):
    st.write(f"**Training Accuracy:** {train_acc:.2f}")
    st.write(f"**Testing Accuracy:** {test_acc:.2f}")

# Sample Prediction Interface
st.subheader("🧪 Test a Single Transaction")
random_idx = st.button("Use Random Test Transaction")

if random_idx:
    index = np.random.randint(0, len(X_test))
    sample = X_test.iloc[index]
    prediction = model.predict(sample.values.reshape(1, -1))[0]
    actual = Y_test.iloc[index]

    st.write("**Sample Input Features:**")
    st.dataframe(pd.DataFrame(sample).T)

    st.markdown(f"**✅ Predicted:** {'Fraud' if prediction == 1 else 'Not Fraud'}")
    st.markdown(f"**🆚 Actual:** {'Fraud' if actual == 1 else 'Not Fraud'}")

# Upload CSV and predict
st.subheader("📁 Upload CSV for Bulk Prediction")
uploaded_file = st.file_uploader("Upload a CSV file with same structure as input features", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    preds = model.predict(input_df)
    input_df['Prediction'] = ['Fraud' if p == 1 else 'Not Fraud' for p in preds]
    st.success("Predictions complete!")
    st.dataframe(input_df)

    csv = input_df.to_csv(index=False).encode()
    st.download_button("📥 Download Results", data=csv, file_name="predictions.csv", mime='text/csv')
