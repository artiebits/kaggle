import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Load the data, and separate the target
iowa_file_path = "../data/home-data-for-ml-course/iowa_train_data.csv"
home_data = pd.read_csv(iowa_file_path)
y = home_data.SalePrice

# Create X (After completing the exercise, you can return to modify this line!)
features = [
    "MSSubClass",
    "LotArea",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "TotRmsAbvGrd",
    "Fireplaces",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
    "MiscVal",
    "MoSold",
    "YrSold",
]

# Select columns corresponding to features, and preview the data
X = home_data[features]

# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# Define the XGBoost model
xgb_model = xgb.XGBRegressor(random_state=1)
xgb_model.fit(train_X, train_y)
xgb_val_predictions = xgb_model.predict(val_X)
xgb_val_mae = mean_absolute_error(xgb_val_predictions, val_y)

print("Validation MAE for XGBoost model: {:,.0f}".format(xgb_val_mae))
