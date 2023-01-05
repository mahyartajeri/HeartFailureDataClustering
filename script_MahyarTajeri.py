from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("./heart_failure_clinical_records_dataset.csv")
print(data.shape)

data.dropna(inplace=True)
print(data.shape)

data.drop("time", inplace=True, axis=1)
data.drop("DEATH_EVENT", inplace=True, axis=1)
print(data.shape)

scaler = RobustScaler()
scaled_data = scaler.fit_transform(data)

print("-------")
print(scaled_data)
print("-------")

pca = PCA()
pca.fit(scaled_data)

plt.figure(figsize=(10, 8))
plt.plot(
    range(1, 12), pca.explained_variance_ratio_.cumsum(), marker="o", linestyle="--"
)
plt.title("Sum of Explained Variance vs. # of Components")
plt.xlabel("# of Components")
plt.ylabel("Culmulative Explained Variance")


pca4 = PCA(n_components=4)


pca4.fit(scaled_data)
pca_data = pca4.transform(scaled_data)

print(pca_data)

plt.figure(figsize=(10, 8))
inertia = []
N = range(1, 21)
for i in N:
    kmeans = KMeans(n_clusters=i, random_state=42).fit(pca_data)
    # print(kmeans.labels_)
    inertia.append(kmeans.inertia_)

plt.plot(N, inertia, "bx-")
plt.title("Inertia vs. Number of Clusters")
plt.xlabel("k")
plt.ylabel("Inertia")

# Chosen 5 clusters based on elbow of the graph.
kmeans5 = KMeans(n_clusters=5, random_state=42).fit(pca_data)


final_data = pd.concat([data.reset_index(drop=True), pd.DataFrame(pca_data)], axis=1)
final_data.columns.values[11:] = ["PCA 1", "PCA 2", "PCA 3", "PCA 4"]
final_data["Cluster Number"] = kmeans5.labels_


print(final_data.head(30))

# Visualize clustering
c = ["hotpink", "red", "blue", "purple", "yellow"]
clusterFig = plt.figure(figsize=(10, 7))
for i in range(len(final_data)):
    plt.scatter(
        final_data["serum_creatinine"][i],
        final_data["ejection_fraction"][i],
        marker="o",
        color=c[final_data["Cluster Number"][i]],
    )
plt.title("Ejection Fraction vs. Serum Creatinine")
plt.xlabel("Serum Creatinine")
plt.ylabel("Ejection Fraction")
clusterFig.canvas.draw()

sns.pairplot(final_data.loc[:, "PCA 1":"Cluster Number"], hue="Cluster Number")

plt.show()
