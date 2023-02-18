import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

melbourne_file_path = "../data/home-data-for-ml-course/melbourne_data.csv"
data = pd.read_csv(melbourne_file_path)

data = data.dropna(axis=0)

y = data.Price

features = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]
X = data[features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Define a random forest model
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(train_X, train_y)
rf_val_predictions = rf_model.predict(val_X)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

mae = mean_absolute_error(val_y, rf_val_predictions)
mse = mean_squared_error(val_y, rf_val_predictions)
r2 = r2_score(val_y, rf_val_predictions)

print("Validation for Random Forest Model.")
print("Mean Absolute Error:", mae)
print("Mean Squared Error:", mse)
print("R-Squared:", r2)
