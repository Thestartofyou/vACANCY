# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load your dataset with relevant features (e.g., town population, unemployment rate, etc.)
# Replace 'data.csv' with the actual dataset path and adjust columns accordingly
data = pd.read_csv('data.csv')
X = data[['Population', 'Unemployment_Rate', 'Median_Income', ...]]  # Include relevant features
y = data['Vacancy_Rate']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Linear Regression model (you can use other regression models as well)
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance (you can use other metrics as needed)
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Now, you can use the trained model to make predictions for rental vacancy rates in specific towns in NH.
# You'll need to provide the same set of features as used during training.

# Example: Predict vacancy rate for a new town
new_town_features = np.array([[population, unemployment_rate, median_income, ...]])  # Replace with actual values
vacancy_rate_prediction = model.predict(new_town_features)
print(f'Predicted Vacancy Rate for the New Town: {vacancy_rate_prediction[0]}')
