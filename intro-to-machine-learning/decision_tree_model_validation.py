import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

melbourne_file_path = "../data/home-data-for-ml-course/melbourne_data.csv"
melbourne_data = pd.read_csv(melbourne_file_path)

melbourne_data = melbourne_data.dropna(axis=0)

y = melbourne_data.Price

melbourne_features = ["Rooms", "Bathroom", "Landsize", "Lattitude", "Longtitude"]
X = melbourne_data[melbourne_features]

melbourne_model = DecisionTreeRegressor(random_state=1)

melbourne_model.fit(X, y)

# Wrong implementation of calculating the mean absolute error (MAE)
predicted_home_prices = melbourne_model.predict(X)
mea = mean_absolute_error(y, predicted_home_prices)
print("Wrong mean absolute error is:", mea)

# We used a single "sample" of houses for both building the model and evaluating it.
# However, evaluating a machine learning model's performance on its training data is a common mistake and
# is known as "overfitting." Since models' practical value come from making predictions on new data,
# we measure performance on data that wasn't used to build the model.
# The most straightforward way to do this is to exclude some data from the model-building process,
# and then use those to test the model's accuracy on data it hasn't seen before

# The scikit-learn library has a function train_test_split to break up the data into two pieces.
# Split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we run this script.
train_X, validation_X, train_y, validation_y = train_test_split(X, y, random_state=0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# Get predicted prices on validation data
val_predictions = melbourne_model.predict(validation_X)
mea = mean_absolute_error(validation_y, val_predictions)
print("The mean absolute error is:", mea)

# The mean absolute error for the in-sample data was about 1k dollars. Out-of-sample it is more than 250,000 dollars.
# This is the difference between a model that is almost exactly right, and
# one that is unusable for most practical purposes. As a point of reference,
# the average home value in the validation data is 1.1 million dollars.
# So the error in new data is about a quarter of the average home value.
# There are many ways to improve this model, such as experimenting to find better features or different model.
