# import the packages 
from sklearn.linear_model import LogisticRegression
import numpy as np 

# Take input 
X = np.array([[2],[3],[4],[5],[7],[9]])

Y = np.array([0,0,0,1,1,1])

# train the model 
model = LogisticRegression()
model.fit(X,Y)

#predict 
new_stu = np.array([[9]])
pred = model.predict(new_stu)
print("The actual value is:", pred)
