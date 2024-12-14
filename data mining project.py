import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples

#Stratification
data = pd.read_csv('Data_mining_assignment.csv')

print(data.head())

stratification_column = 'SEX'


train_data, test_data = train_test_split(data, test_size=0.3,
 stratify=data[stratification_column], random_state=42)


print("Training Set Distribution:")
print(train_data[stratification_column].value_counts(normalize=True))

print("Test Set Distribution:")
print(test_data[stratification_column].value_counts(normalize=True))


 #Pie Chart for SEX
sex_counts = data['SEX'].value_counts()
sex_counts.plot.pie(autopct='%1.1f%%', figsize=(6, 6), title="Distribution of SEX")
plt.show()

 #Pie Chart for RACE
sex_counts = data['RACE'].value_counts()
sex_counts.plot.pie(autopct='%1.1f%%', figsize=(6, 6), title="Distribution of RACE")
plt.show()

# Pie Chart for ANYMEDS
sex_counts = data['ANYMEDS'].value_counts()
sex_counts.plot.pie(autopct='%1.1f%%', figsize=(6, 6), title="Distribution of ANYMEDS")
plt.show()

pd.crosstab(data['SEX'], data['RACE']).plot(kind='bar', stacked=True, figsize=(10, 6))
plt.title("Distribution of RACE by SEX")
plt.ylabel("Count")
plt.show()

sns.countplot(x="SEX", data=train_data)
plt.title("SEX Distribution in Training Set")
plt.show()

sns.countplot(x="SEX", data=test_data)
plt.title("SEX Distribution in Test Set")
plt.show()


train_sex_dist = train_data['SEX'].value_counts()
test_sex_dist = test_data['SEX'].value_counts()

# Pie Chart for Training Set
train_sex_dist.plot.pie(autopct='%1.1f%%', figsize=(6, 6), title="SEX Distribution in Training Set")
plt.show()

 #Pie Chart for Test Set
test_sex_dist.plot.pie(autopct='%1.1f%%', figsize=(6, 6), title="SEX Distribution in Test Set")
plt.show()



#Clustering

features = ['SEX', 'RACE', 'ANYMEDS', 'NACCFAM', 'SMOKYRS']  


X = data[features]


scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)


wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_normalized)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Curve
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method to Determine Optimal k')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('WCSS')
plt.show()


k = 3  
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(X_normalized)


data['Cluster'] = clusters

# Reduce to 2 dimensions using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_normalized)

# Create a DataFrame for visualization
pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
pca_df['Cluster'] = clusters

# Scatter plot of clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2')
plt.title('Clusters Visualized Using PCA')
plt.show()


# Calculate mean values for each feature by cluster
cluster_summary = data.groupby('Cluster').mean()
print("Cluster Summary:")
print(cluster_summary)

# Visualize distributions for each feature by cluster
sns.boxplot(data=data, x='Cluster', y='SMOKYRS', palette='Set2')
plt.title('Distribution of SMOKYRS by Cluster')
plt.show()

# Visualize distributions for each feature by cluster
sns.boxplot(data=data, x='Cluster', y='RACE', palette='Set2')
plt.title('Distribution of RACE by Cluster')
plt.show()

# Visualize distributions for each feature by cluster
sns.boxplot(data=data, x='Cluster', y='NACCFAM', palette='Set2')
plt.title('Distribution of NACCFAM by Cluster')
plt.show()

# Visualize distributions for each feature by cluster
sns.boxplot(data=data, x='Cluster', y='ANYMED', palette='Set2')
plt.title('Distribution of ANYMED by Cluster')
plt.show()

# KMeans model
kmeans = KMeans(n_clusters=3, random_state=42) 
clusters = kmeans.fit_predict(data)

# Calculate silhouette score
score = silhouette_score(data, clusters)
print(f"Silhouette Score: {score:.2f}")

silhouette_vals = silhouette_samples(data, clusters)

# Plot silhouette analysis
plt.figure(figsize=(10, 6))
y_lower, y_upper = 0, 0
for i in range(3):  
    cluster_silhouette_vals = silhouette_vals[clusters == i]
    cluster_silhouette_vals.sort()
    y_upper += len(cluster_silhouette_vals)
    plt.barh(range(y_lower, y_upper), cluster_silhouette_vals, height=1)
    y_lower += len(cluster_silhouette_vals)

plt.axvline(score, color="red", linestyle="--") 
plt.title("Silhouette Analysis")
plt.xlabel("Silhouette Coefficient")
plt.ylabel("Cluster")
plt.show()