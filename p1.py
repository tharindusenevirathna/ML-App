import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, silhouette_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

# Task 1: Project Overview and Demo
st.title("Mushroom Classification and Clustering Web App")
st.sidebar.title("Mushroom Web App")
st.write("""
### This app allows you to classify mushrooms as edible or poisonous using various machine learning algorithms, 
as well as perform clustering on the dataset.
""")

# Step 1: Select Type of Problem
problem_type = st.sidebar.selectbox("Select Type of Problem", ["Supervised Learning", "Unsupervised Learning"])

# Step 2: Add Dataset
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    label = LabelEncoder()
    for col in df.columns:
        df[col] = label.fit_transform(df[col])
    
    # Show dataset preview
    if st.sidebar.checkbox("Show Raw Data", False):
        st.subheader("Dataset Preview")
        st.write(df)

    if problem_type == "Supervised Learning":
        # Define features and target for supervised learning
        X = df.drop('type', axis=1)  # Replace 'type' with your target column
        y = df['type']  # Replace 'type' with your target column
        
        # Task 4: Creating Training and Test Sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Sidebar for choosing classifier
        st.sidebar.header("Choose Classifier")
        classifier_name = st.sidebar.selectbox("Select Classifier", ["SVM", "Logistic Regression", "Random Forest", "XGBoost", "K-Nearest Neighbors", "Decision Tree", "Naive Bayes"])

        # Hyperparameter tuning options based on selected classifier
        if classifier_name == "SVM":
            st.sidebar.header("SVM Hyperparameter Tuning")
            C = st.sidebar.slider("C (Regularization Parameter)", 0.01, 100.0, 1.0)
            kernel = st.sidebar.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
            gamma = st.sidebar.selectbox("Gamma", ["scale", "auto"] + [0.01, 0.1, 1, 10])
            degree = st.sidebar.slider("Degree (for Polynomial Kernel)", 2, 5, 3)

        elif classifier_name == "Logistic Regression":
            st.sidebar.header("Logistic Regression Hyperparameter Tuning")
            C_lr = st.sidebar.slider("C (Regularization Parameter)", 0.01, 100.0, 1.0)
            solver = st.sidebar.selectbox("Solver", ["lbfgs", "liblinear", "sag", "saga", "newton-cg"])
            max_iter = st.sidebar.slider("Max Iterations", 100, 1000, 100)

        elif classifier_name == "Random Forest":
            st.sidebar.header("Random Forest Hyperparameter Tuning")
            n_estimators = st.sidebar.slider("Number of Trees", 10, 200, 100)
            max_depth = st.sidebar.slider("Max Depth", 1, 20, 10)
            min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)

        elif classifier_name == "K-Nearest Neighbors":
            st.sidebar.header("K-Nearest Neighbors Hyperparameter Tuning")
            n_neighbors = st.sidebar.slider("Number of Neighbors", 1, 50, 5)
            weights = st.sidebar.selectbox("Weights", ["uniform", "distance"])
            algorithm = st.sidebar.selectbox("Algorithm", ["auto", "ball_tree", "kd_tree", "brute"])

        elif classifier_name == "Decision Tree":
            st.sidebar.header("Decision Tree Hyperparameter Tuning")
            max_depth = st.sidebar.slider("Max Depth", 1, 20, None)
            min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)

        elif classifier_name == "Naive Bayes":
            st.sidebar.header("Naive Bayes Hyperparameter Tuning")
            # No hyperparameters to tune for basic Naive Bayes, can include options for Gaussian, Multinomial, etc.

        elif classifier_name == "XGBoost":
            st.sidebar.header("XGBoost Hyperparameter Tuning")
            n_estimators_xgb = st.sidebar.slider("Number of Trees", 10, 200, 100)
            learning_rate = st.sidebar.slider("Learning Rate", 0.01, 1.0, 0.1)
            max_depth_xgb = st.sidebar.slider("Max Depth", 1, 20, 6)

        # Button to train the selected classifier
        if st.sidebar.button("Train Classifier"):
            if classifier_name == "SVM":
                svc = SVC(C=C, kernel=kernel, gamma=gamma, degree=degree)
                svc.fit(X_train, y_train)
                y_pred = svc.predict(X_test)

            elif classifier_name == "Logistic Regression":
                lr = LogisticRegression(C=C_lr, solver=solver, max_iter=max_iter)
                lr.fit(X_train, y_train)
                y_pred = lr.predict(X_test)

            elif classifier_name == "Random Forest":
                rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=42)
                rf.fit(X_train, y_train)
                y_pred = rf.predict(X_test)

            elif classifier_name == "K-Nearest Neighbors":
                knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)

            elif classifier_name == "Decision Tree":
                dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
                dt.fit(X_train, y_train)
                y_pred = dt.predict(X_test)

            elif classifier_name == "Naive Bayes":
                nb = GaussianNB()
                nb.fit(X_train, y_train)
                y_pred = nb.predict(X_test)

            elif classifier_name == "XGBoost":
                xgb = XGBClassifier(n_estimators=n_estimators_xgb, learning_rate=learning_rate, max_depth=max_depth_xgb, use_label_encoder=False, eval_metric='mlogloss')
                xgb.fit(X_train, y_train)
                y_pred = xgb.predict(X_test)

            # Display Evaluation Metrics
            st.subheader(f"Evaluation Metrics for {classifier_name}")
            st.write("### Classification Report")
            report = classification_report(y_test, y_pred)
            st.text(report)

            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)

    elif problem_type == "Unsupervised Learning":
        # Define features for unsupervised learning
        X = df  # Use the entire dataset for clustering
        
        # Sidebar for choosing clustering algorithm
        st.sidebar.header("Choose Clustering Algorithm")
        clustering_name = st.sidebar.selectbox("Select Clustering Algorithm", ["K-Means", "Hierarchical Clustering"])

        # Hyperparameter tuning options for K-Means
        if clustering_name == "K-Means":
            st.sidebar.header("K-Means Hyperparameter Tuning")
            n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)

            # Button to train K-Means
            if st.sidebar.button("Train K-Means"):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                kmeans.fit(X)
                labels = kmeans.labels_
                st.subheader("K-Means Clustering Results")
                
                # Adding cluster labels to the dataframe
                df['Cluster'] = labels
                
                # Show results
                st.write(df)

                # Silhouette Score
                silhouette_avg = silhouette_score(X, labels)
                st.write(f"Silhouette Score: {silhouette_avg:.2f}")

                # PCA for visualization
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X)
                df_pca = pd.DataFrame(data=X_pca, columns=["Component 1", "Component 2"])
                df_pca['Cluster'] = labels

                # Visualization
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=df_pca, x="Component 1", y="Component 2", hue="Cluster", palette="Set1")
                st.pyplot(plt)

        # Hyperparameter tuning options for Hier
