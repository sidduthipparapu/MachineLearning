from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Features:
# [words_per_day, initiated_chat, emojis_per_day, avg_reply_time]

X = [
    [500, 1, 40, 2],
    [450, 1, 35, 3],
    [550, 1, 50, 2],
    [480, 1, 45, 4],

    [250, 0, 20, 15],
    [300, 0, 15, 12],
    [270, 1, 25, 10],
    [220, 0, 18, 14],

    [50, 0, 2, 120],
    [80, 0, 3, 100],
    [40, 0, 1, 150],
    [60, 0, 2, 130]
]

# 1. Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Create K-Means model
model = KMeans(n_clusters=3, random_state=42, n_init=10)

# 3. Train the model
model.fit(X_scaled)

# 4. Get cluster labels
labels = model.labels_

print("Cluster labels:")
print(labels)

# 5. Plot the clusters
reply_time = [row[3] for row in X]
words = [row[0] for row in X]

plt.scatter( reply_time , words, c=labels)

plt.xlabel("Average Reply Time (minutes)")
plt.ylabel("Words per Day")
plt.title("Communication Behavior Clusters")

plt.show()