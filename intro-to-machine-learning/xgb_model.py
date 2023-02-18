import pandas as pd
import xgboost as xgb

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

# Define the XGBoost model
xgb_model = xgb.XGBRegressor(random_state=1)
xgb_model.fit(X, y)
xgb_val_predictions = xgb_model.predict(X)
