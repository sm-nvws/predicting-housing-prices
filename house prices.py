import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # Scaling the data
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


df = pd.read_csv("/Users/sameermahmud/Downloads/NY-House-Dataset.csv")


X = df[['PROPERTYSQFT']]  
y = df['PRICE']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


model = Sequential()


model.add(Dense(64, input_dim=1, activation='relu'))  
model.add(Dense(32, activation='relu')) 
model.add(Dense(1))  


model.compile(optimizer='adam', loss='mean_squared_error')


history = model.fit(X_train_scaled, y_train, epochs=100, validation_split=0.2)


y_pred = model.predict(X_test_scaled)


y_pred_rescaled = y_pred.flatten()


comparison = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred_rescaled})
print(comparison.head())

plt.scatter(X_test, y_test, color='blue', label='Actual')  # Actual data points
plt.scatter(X_test, y_pred_rescaled, color='red', label='Predicted')  # Predicted data points
plt.xlabel('Size (sq ft)')
plt.ylabel('Price ($)')
plt.title('Size vs Price Prediction (TensorFlow Model)')
plt.legend()
plt.show()
