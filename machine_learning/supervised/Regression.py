from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Data
X = [[1], [2], [3], [4], [5]]
y = [30, 40, 50, 60, 70]

# Create and train model
model = LinearRegression()
model.fit(X, y)

# Prediction
prediction = model.predict([[6]])
print("Predicted salary:", prediction[0])

# Plot actual data
plt.scatter(X, y)

# Plot regression line
plt.plot(X, model.predict(X))

plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.title("Linear Regression")

plt.show()