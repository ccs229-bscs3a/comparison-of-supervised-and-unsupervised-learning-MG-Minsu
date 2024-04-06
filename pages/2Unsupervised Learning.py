#Input the relevant libraries
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, metrics
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.cluster import KMeans
import time

# Define the Streamlit app
def app():

    st.subheader('K-means clustering applied to the diabetes Dataset')
    


    if st.button("Begin"):
        # Load the diabetes dataset
        diabetes = datasets.load_diabetes()
        X = diabetes.data  # Features
        y = diabetes.target  # Target labels (species)

        # Define the K-means model with 3 clusters (known number of species)
        kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)

        # Train the K-means model
        kmeans.fit(X)

        # Get the cluster labels for the data
        y_kmeans = kmeans.labels_

        # Since there are no true labels for unsupervised clustering,
        # we cannot directly calculate accuracy.
        # We can use silhouette score to evaluate cluster separation

        # Calculate WCSS
        wcss = kmeans.inertia_
        st.write("Within-Cluster Sum of Squares:", wcss)

        silhouette_score = metrics.silhouette_score(X, y_kmeans)
        st.write("K-means Silhouette Score:", silhouette_score)

        text = """**Within-Cluster Sum of Squares (WCSS): 78.85144142614598**
        This value alone doesn't tell the whole story. A lower WCSS generally indicates tighter 
        clusters, but it depends on the scale of your data and the number of clusters used (k).
        \n**K-mmeans Silhouette Score: 0.5528190123564095**
        * This score provides a more interpretable measure of cluster quality. It 
        ranges from -1 to 1, where:
        * Values closer to 1 indicate well-separated clusters.
        * Values around 0 suggest clusters are indifferently assigned (data points could belong to either cluster).
        * Negative values indicate poorly separated clusters (data points in a cluster are closer to points in other clusters).
        In this case, a Silhouette Score of 0.5528 suggests:
        * **Moderately separated clusters:** The data points within a cluster are somewhat closer to their centroid than to centroids of other clusters. There's some separation, but it's not perfect
        * **Potential for improvement:** You might consider exploring different numbers of clusters (k) or using different initialization methods for K-means to see if a better clustering solution can be achieved with a higher Silhouette Score (closer to 1).
        * The diabetes dataset is relatively well-separated into three flower species. A Silhouette Score above 0.5 might be achievable with an appropriate number of clusters (k=3) and good initialization.
        * The optimal k can vary depending on the specific dataset and the desired level of granularity in the clustering."""
        with st.expander("Click here for more information."):\
            st.write(text)
            
        # Get predicted cluster labels
        y_pred = kmeans.predict(X)

        # Get unique class labels and color map
        unique_labels = list(set(y_pred))
        colors = plt.cm.get_cmap('viridis')(np.linspace(0, 1, len(unique_labels)))

        
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

       


#run the app
if __name__ == "__main__":
    app()
