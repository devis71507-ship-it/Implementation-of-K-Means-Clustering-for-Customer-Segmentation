# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Collect and load the customer dataset containing features like income, age, and spending score.

2. Choose the number of clusters (K) and initialize (K) centroids randomly.

3. Assign each customer to the nearest centroid using distance calculation (usually Euclidean distance).

4. Recalculate the centroid of each cluster by taking the mean of all data points in that cluster.

5. Repeat the assignment and centroid update steps until the centroids no longer change, then analyze the formed customer segments.

## Program:

/*
Program to implement the K Means Clustering for Customer Segmentation.

Developed by: S.DEVI

RegisterNumber:  212225100008
*/
```
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

data = pd.read_csv(r"C:\Users\sdhan\Downloads\Mall_Customers.csv")

print(data.head())
print(data.columns)

features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = data[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia_values = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)  # Explicit n_init to suppress warning
    kmeans.fit(X_scaled)
    inertia_values.append(kmeans.inertia_)
    
plt.figure(figsize=(8, 4))
plt.plot(range(1, 11), inertia_values, marker='o', linestyle='-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

optimal_clusters = 4
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)  # Explicit n_init
kmeans.fit(X_scaled)

data['Cluster'] = kmeans.labels_

sil_score = silhouette_score(X_scaled, kmeans.labels_)
print(f'Silhouette Score: {sil_score}')

print("\nName: DEVI S")
print("Reg No.: 212225100008\n")
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data,x='Annual Income (k$)',y='Spending Score (1-100)',hue='Cluster', palette='viridis',s=100,alpha=0.7)
plt.title('Customer Segmentation based on Annual Income and Spending Score')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()
```

## Output:
<img width="661" height="169" alt="image" src="https://github.com/user-attachments/assets/4c571e89-68a8-4acf-967a-488bfbae13bd" />

<img width="733" height="390" alt="image" src="https://github.com/user-attachments/assets/b633c4dc-6b5c-4511-a82c-7c39a8c614f2" />

<img width="928" height="566" alt="image" src="https://github.com/user-attachments/assets/a2dd9e3d-9da6-469e-b5d6-f5de4a8c73ea" />

## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
