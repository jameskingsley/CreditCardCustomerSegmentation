import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# -- Helper functions --

def remove_outliers_iqr(df, columns, factor=1.5):
    df = df.copy()
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    return df

def map_clusters(cluster_series):
    cluster_labels_map = {
        0: "Inactive Users",
        1: "Cash Users",
        2: "High Spenders",
        3: "Revolvers"
    }
    return cluster_series.map(cluster_labels_map)

# -- Streamlit UI --

st.title("Credit Card Customer Segmentation & Prediction")

uploaded_file = st.file_uploader("Upload your CSV data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data loaded successfully. Sample:")
    st.dataframe(df.head())

    # Drop rows with missing values
    df = df.dropna()
    st.write(f"Data shape after dropping missing rows: {df.shape}")

    # Select numeric columns for clustering
    num_cols = ['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS']

    # Remove outliers
    df_clean = remove_outliers_iqr(df, num_cols)
    st.write(f"Data shape after outlier removal: {df_clean.shape}")

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean[num_cols])

    # --- Elbow Method ---
    st.subheader("Elbow Method to find optimal k")
    max_k = 10
    inertias = []
    for k in range(1, max_k+1):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(X_scaled)
        inertias.append(km.inertia_)

    fig_elbow, ax_elbow = plt.subplots()
    ax_elbow.plot(range(1, max_k+1), inertias, marker='o')
    ax_elbow.set_xlabel("Number of clusters (k)")
    ax_elbow.set_ylabel("Inertia")
    ax_elbow.set_title("Elbow Method For Optimal k")
    st.pyplot(fig_elbow)

    # User selects k
    optimal_k = st.slider("Select number of clusters (k)", min_value=2, max_value=10, value=4)

    # Fit final KMeans
    kmeans = KMeans(n_clusters=optimal_k, random_state=42)
    cluster_labels = kmeans.fit_predict(X_scaled)

    # Add cluster labels to dataframe
    df_clean['Cluster'] = cluster_labels

    # Map cluster names if k=4, else just show cluster number
    if optimal_k == 4:
        df_clean['Segment'] = map_clusters(df_clean['Cluster'])
    else:
        df_clean['Segment'] = "Cluster " + df_clean['Cluster'].astype(str)

    # Display silhouette score
    sil_score = silhouette_score(X_scaled, cluster_labels)
    st.write(f"Silhouette Score: **{sil_score:.3f}**")

    # Cluster size distribution plot
    st.subheader("Cluster Size Distribution")
    fig_size, ax_size = plt.subplots()
    df_clean['Segment'].value_counts().plot(kind='bar', color='coral', ax=ax_size)
    ax_size.set_ylabel("Number of Customers")
    ax_size.set_xlabel("Segment")
    ax_size.set_title("Cluster Size Distribution")
    st.pyplot(fig_size)

    # Cluster profile report
    st.subheader("Cluster Profile (Average feature values)")
    cluster_profile = df_clean.groupby('Segment').mean(numeric_only=True).round(2)

    # Feature importance: std dev of mean feature values across clusters
    feature_variances = cluster_profile.std().sort_values(ascending=False)

    st.write(cluster_profile)

    # Feature importance plot
    st.subheader("Estimated Feature Importance")
    fig_feat, ax_feat = plt.subplots()
    feature_variances.plot(kind='bar', color='teal', ax=ax_feat)
    ax_feat.set_ylabel("Standard Deviation")
    ax_feat.set_title("Feature Importance by Variance Across Clusters")
    ax_feat.set_xticklabels(ax_feat.get_xticklabels(), rotation=45, ha='right')
    st.pyplot(fig_feat)

    # PCA visualization
    st.subheader("PCA Visualization of Clusters")
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(X_scaled)

    fig_pca, ax_pca = plt.subplots(figsize=(10, 7))
    sns.scatterplot(x=pca_components[:, 0], y=pca_components[:, 1], hue=df_clean['Segment'], palette='Set2', ax=ax_pca)
    ax_pca.set_xlabel("PCA 1")
    ax_pca.set_ylabel("PCA 2")
    ax_pca.set_title("PCA Visualization of Customer Segments")
    ax_pca.legend(title='Segment')
    st.pyplot(fig_pca)

    # Download clustered dataset CSV
    csv_data = df_clean.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Clustered Data as CSV",
        data=csv_data,
        file_name='credit_card_customers_segmented.csv',
        mime='text/csv'
    )

    # --- Real-time prediction ---
    st.write("---")
    st.header("Predict Cluster for New Customer")

    with st.form("predict_form"):
        st.write("Enter new customer data for prediction:")
        new_data = {}
        for col in num_cols:
            min_val = float(df_clean[col].min())
            max_val = float(df_clean[col].max())
            mean_val = float(df_clean[col].mean())
            new_data[col] = st.number_input(
                label=col,
                min_value=min_val,
                max_value=max_val,
                value=mean_val,
                format="%.4f"
            )
        submitted = st.form_submit_button("Predict Cluster")

    if submitted:
        new_df = pd.DataFrame([new_data])
        new_scaled = scaler.transform(new_df)
        pred_cluster = kmeans.predict(new_scaled)[0]

        if optimal_k == 4:
            segment_name = map_clusters(pd.Series(pred_cluster)).iloc[0]
        else:
            segment_name = f"Cluster {pred_cluster}"

        st.success(f"Predicted Cluster: {pred_cluster} ({segment_name})")

else:
    st.info("Please upload a CSV file to get started.")
