# CreditCardCustomerSegmentation

 ## Credit Card Customer Segmentation & Real-Time Prediction App
 
Welcome to the Credit Card Customer Segmentation & Real-Time Prediction app — your all-in-one interactive tool to segment credit card customers using KMeans clustering and predict customer segments in real-time!

###### Features
* Data Upload & Cleaning
Upload your credit card customer dataset (CSV) and watch the app automatically clean, remove missing values, and filter outliers for robust clustering.

* Interactive Elbow Plot
Visualize the elbow method to identify the optimal number of clusters in your data.

* Dynamic Clustering
Select the number of clusters k with a slider and instantly view updated clusters, segments, and quality metrics.

* Insightful Cluster Profiles
Explore average feature values per segment and understand what makes each cluster unique.

* Feature Importance Visualization
Discover the most discriminative features driving your customer segments.

* PCA Visualization
Visualize customer clusters in 2D space using Principal Component Analysis for intuitive understanding.

* Real-Time Segment Prediction
Input new customer data on the fly and get instant predictions of their cluster membership.

* Download Results
Export the enriched dataset with cluster labels for further analysis or reporting.

###### Technologies Used
* Python 3

* Streamlit for interactive web app

* Pandas & NumPy for data processing

* Scikit-learn for clustering, scaling, PCA, and evaluation

* Seaborn & Matplotlib for rich visualizations

###### How to Use
Prepare your dataset as a CSV with numeric features including at least:
BALANCE, PURCHASES, CASH_ADVANCE, CREDIT_LIMIT, PAYMENTS.

###### Run the Streamlit app:
* Upload your CSV file via the file uploader.

###### Explore clustering results:

* View elbow plot and select the optimal cluster number.

* Examine cluster profiles, size distributions, and feature importances.

* Visualize clusters with PCA plots.

* Predict new customer segments in real-time by entering their financial data.

* Download the segmented dataset for your records.

###### Why This App?
Customer segmentation is key for targeted marketing, personalized services, and improving retention. This app empowers data scientists and business analysts alike with an intuitive interface, powerful clustering, and instant predictions — no heavy coding required!

###### Project Structure
├── app.py                   
├── credit_card_customers.csv 
├── requirements.txt         
└── README.md                

###### Contributing & Support
Contributions are welcome! 
Feel free to open issues or submit pull requests to improve the app.

If you encounter any issues or need help, contact me at [bestmankingsley001@gmail.com] or open an issue here.

