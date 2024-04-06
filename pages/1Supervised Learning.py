# Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import datasets
import time

# Define the Streamlit app
def app():
    st.subheader('Supervised Learning, Classification, and KNN with Diabetes Dataset')
   
    k = st.sidebar.slider(
        label="Select the value of k:",
        min_value= 2,
        max_value= 10,
        value=5,  # Initial value
    )

    if st.button("Begin"):
        # Load the Diabetes dataset
        diabetes = datasets.load_diabetes()
        X = diabetes.data  # Features
        y = diabetes.target  # Target labels

        # KNN for supervised classification (reference for comparison)
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X, y)

        # Predict the diabetes progression for the patients
        y_pred = knn.predict(X)

        st.write('Confusion Matrix')
        cm = confusion_matrix(y, y_pred)
        st.text(cm)

        st.subheader('Performance Metrics')
        st.text(classification_report(y, y_pred))


        selected_feature_index = 2  
        selected_feature = diabetes.feature_names[selected_feature_index]

        selected_feature_index1 = 3  
        selected_feature1 = diabetes.feature_names[selected_feature_index1]

        # Plotting the selected feature against the target
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X[:, selected_feature_index], y, c='blue', label='Actual Diabetes Progression')
        ax.scatter(X[:, selected_feature_index], y_pred, c='red', label='Predicted Diabetes Progression')
        ax.set_xlabel(selected_feature)
        ax.set_ylabel(selected_feature1)
        ax.set_title(f'Visualization of Diabetes Dataset: {selected_feature} vs  {selected_feature1}')
        ax.legend()
        st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    app()
