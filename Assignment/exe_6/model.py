import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Load the dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/BostonHousing.csv"
df = pd.read_csv(url)
print("Dataset loaded")

# Step 2: Exploratory Data Analysis (EDA)
print("First 5 rows:\n", df.head())
print("\nSummary statistics:\n", df.describe())

# Check for missing values
print("\nMissing values:\n", df.isnull().sum())

# Step 3: Data Visualization
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('/app/figures/correlation_heatmap.png')  # Save figure in container
plt.show()

# Scatter plot between RM and MEDV
plt.figure(figsize=(8, 6))
plt.scatter(df['rm'], df['medv'], color='blue')
plt.title('Rooms vs House Price')
plt.xlabel('Average number of rooms per dwelling (RM)')
plt.ylabel('Median value of homes (MEDV)')
plt.savefig('/app/figures/rm_vs_medv.png')  # Save figure in container
plt.show()

# Step 4: Prepare data for Linear Regression
X = df[['rm']]  # Feature
y = df['medv']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the Model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# Step 8: Visualize the Linear Regression Results
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Actual Prices')
plt.plot(X_test, y_pred, color='red', label='Predicted Prices')
plt.title('Actual vs Predicted House Prices')
plt.xlabel('Average Number of Rooms (RM)')
plt.ylabel('House Price')
plt.legend()
plt.savefig('/app/figures/actual_vs_predicted.png')  # Save figure in container
plt.show()