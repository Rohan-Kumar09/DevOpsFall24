import sqlite3
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px

# Step 1: Connect to the database and fetch data
conn = sqlite3.connect('mydatabase.db')
query = """
SELECT DISTINCT 
    [Industry Title],
    [SIC Code],
    [Company Name],
    [cityba],
    [zipba]
FROM Ticker_IDs
LEFT JOIN SIC_Codes
ON SIC_Codes.[SIC Code] = Ticker_IDs.[sic]
"""
df = pd.read_sql_query(query, conn)
conn.close()

# Step 2: Preprocess the data
# Drop rows with missing values for clustering purposes
df_cleaned = df.dropna(subset=['Industry Title', 'cityba', 'zipba'])

# Encode categorical data
label_encoder = LabelEncoder()
df_cleaned['Industry_Encoded'] = label_encoder.fit_transform(df_cleaned['Industry Title'])
df_cleaned['City_Encoded'] = label_encoder.fit_transform(df_cleaned['cityba'])

# Combine features for clustering
X = df_cleaned[['Industry_Encoded', 'City_Encoded']]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Perform K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df_cleaned['Cluster'] = kmeans.fit_predict(X_scaled)

# Step 4: Visualize the clusters interactively using Plotly
fig = px.scatter(
    df_cleaned, 
    x='Industry_Encoded', 
    y='City_Encoded', 
    color='Cluster', 
    hover_data=['Industry Title', 'cityba', 'Company Name'],
    title='Interactive Company Clusters by Industry and Location'
)
fig.update_layout(
    xaxis_title='Industry (Encoded)',
    yaxis_title='City (Encoded)',
    coloraxis_colorbar=dict(title="Cluster")
)
fig.show()

# Step 5: Export results
df_cleaned.to_csv('clustered_companies.csv', index=False)
print("Clustering completed. Results saved to 'clustered_companies.csv'.")
