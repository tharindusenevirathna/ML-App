import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Task 1: Project Overview and Demo
st.title("Mushroom Classification Web App")
st.sidebar.title("Mushroom Classification Web App")
st.write("""
### This app allows you to classify mushrooms as edible or poisonous using various machine learning algorithms. 
You can explore the dataset, visualize evaluation metrics, and interactively train classifiers.
""")

# Task 2: Load the Mushrooms Data Set
@st.cache_data
def load_data():
    data = pd.read_csv("mushrooms.csv")
    label = LabelEncoder()
    for col in data.columns:
        data[col] = label.fit_transform(data[col])
    return data

df = load_data()

if st.sidebar.checkbox("Show Raw Data", False):
    st.subheader("Mushrooms Dataset")
    st.write(df)

# Define features and target
def split_data(df):
    X = df.drop('type', axis=1)  # Features
    y = df['type']  # Target (edible/poisonous)
    # Task 4: Creating Training and Test Sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_data(df)

# Classifier selection
classifier_options = {
    "Support Vector Classifier": SVC(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

# Sidebar buttons
classifier_name = st.sidebar.selectbox("Select Classifier", list(classifier_options.keys()))
metrics_options = ['Accuracy', 'Confusion Matrix', 'Classification Report']
selected_metrics = st.sidebar.multiselect("Select Metrics to Display", metrics_options)

# Train and Evaluate the Selected Classifier
if st.sidebar.button("Show Metrics"):
    model = classifier_options[classifier_name]
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if 'Accuracy' in selected_metrics:
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f"### {classifier_name} Results")
        st.write(f"**Accuracy:** {accuracy:.2f}")

    if 'Confusion Matrix' in selected_metrics:
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        ConfusionMatrixDisplay(confusion_matrix=cm).plot(ax=ax)
        st.pyplot(fig)

    if 'Classification Report' in selected_metrics:
        st.subheader("Classification Report")
        report = classification_report(y_test, y_pred)
        st.text(report)  # Displaying the report as plain text

# Compare all classifiers
if st.sidebar.button("Compare Classifiers"):
    results = {}
    for name, model in classifier_options.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Store results
        results[name] = {
            "Accuracy": accuracy,
            "Precision": report['1']['precision'],
            "Recall": report['1']['recall'],
            "F1-Score": report['1']['f1-score']
        }
    
    # Display the results in a DataFrame
    results_df = pd.DataFrame(results).T
    st.subheader("Classifier Comparison Results")
    st.write(results_df)
