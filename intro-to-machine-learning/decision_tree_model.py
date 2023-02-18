import pandas as pd
from sklearn.tree import DecisionTreeRegressor

melbourne_file_path = "../data/home-data-for-ml-course/melbourne_data.csv"
melbourne_data = pd.read_csv(melbourne_file_path)

melbourne_data = melbourne_data.dropna(axis=0)

# Select the column we want to predict, which is called the prediction target.
# By convention, the prediction target is called y.
y = melbourne_data.Price

# The columns that will be inputted into our model (and later used to make predictions) are called "features."
# Sometimes, you will use all columns except the target as features.
# Other times you'll be better off with fewer features.
melbourne_features = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]
# By convention, this data is called X.
X = melbourne_data[melbourne_features]

# Define model. Specify a number for random_state to ensure same results each run
# Many machine learning models allow some randomness in model training.
# Specifying a number for random_state ensures you get the same results in each run. This is considered a good practice.
# You use any number, and model quality won't depend meaningfully on exactly what value you choose.
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)

# Predict
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are:")
print(melbourne_model.predict(X.head()))
