import pandas as pd
from sklearn.ensemble import RandomForestRegressor

iowa_file_path = "../data/home-data-for-ml-course/iowa_train_data.csv"
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice

# Create X (After completing the exercise, you can return to modify this line!)
features = [
    "LotArea",
    "YearBuilt",
    "1stFlrSF",
    "2ndFlrSF",
    "FullBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
]

# Select columns corresponding to features, and preview the data
X = home_data[features]

# read test data file using pandas
test_data_path = "data/iowa_test_data.csv"
test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]

# Define a random forest model
rf_model_on_full_data = RandomForestRegressor(random_state=1)
rf_model_on_full_data.fit(X, y)
test_preds = rf_model_on_full_data.predict(test_X)
