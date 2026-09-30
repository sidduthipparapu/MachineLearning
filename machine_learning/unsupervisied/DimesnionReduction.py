#It is done by PCA
# PCA called as principal Component Analysis 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Features
X = [
    [500, 1, 40, 2],
    [450, 1, 35, 3],
    [550, 1, 50, 2],
    [480, 1, 45, 4],

    [250, 0, 20, 15],
    [300, 0, 15, 12],
    [280, 1, 25, 10],
    [220, 0, 18, 14],

    [50, 0, 2, 120],
    [80, 0, 3, 100],
    [40, 0, 1, 150],
    [60, 0, 2, 130]
]

# Step 1: Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Create PCA
pca = PCA(n_components=2)

# Step 3: Transform 4 dimensions → 2 dimensions
X_pca = pca.fit_transform(X_scaled)

print("Original shape:", len(X), "x", len(X[0]))
print("PCA shape:", X_pca.shape)

# Step 4: Plot
plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1]
)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA Visualization")

plt.show()