# prompt: import basic libarraies

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df2 = pd.read_csv('df1.csv')
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train2 = pd.merge(train[['id_seqpos']], df2 , on = 'id_seqpos')
test2 = pd.merge(test[['id_seqpos']], df2 , on = 'id_seqpos')
train2.drop(['id_seqpos', 'id_number'], axis=1, inplace=True)
test2.drop(['id_seqpos', 'id_number'], axis=1, inplace=True)

df2.info()

train.info()

train2.info()

test2.info()

test2.isnull().sum()

train2.head()

# prompt: show all columns fo train2

import pandas as pd
pd.set_option('display.max_columns', None)
train2.head()




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error

# Load your train2 and test2 datasets (replace with your actual file paths or DataFrame objects)
# train2 = pd.read_csv('path_to_train2.csv')
# test2 = pd.read_csv('path_to_test2.csv')

# Assuming 'Reactivity', 'deg_Mg_pH10', and 'deg_Mg_50C' are the target variables
target_columns = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']

# Separate features and target variables
X = train2.drop(target_columns, axis=1)
y = train2[target_columns]

# Encode categorical features in the training set
label_encoders = {}
for column in X.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    X[column] = label_encoders[column].fit_transform(X[column])

# Encode categorical features in the test set
X_test = test2.drop(target_columns, axis=1, errors='ignore')  # Ignore if target columns don't exist in test2
for column in X_test.select_dtypes(include=['object']).columns:
    if column in label_encoders:  # Use the same label encoders as the training set
        X_test[column] = label_encoders[column].transform(X_test[column])
    else:
        label_encoders[column] = LabelEncoder()
        X_test[column] = label_encoders[column].fit_transform(X_test[column])

# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid for XGBoost
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

# Initialize the XGBoost regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')

# Perform GridSearchCV to find the best hyperparameters
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best model and parameters
best_xgb_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# Predict on the validation set
y_pred = best_xgb_model.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred, multioutput='raw_values'))
mean_rmse = np.mean(rmse)
print(f'XGBoost RMSE: {rmse}')
print(f'Best Params: {best_params}')

# Retrain the best model on the entire train2 dataset
best_xgb_model.fit(X, y)

# Use the best model to predict on the test set
X_test = test2.drop(['Reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'], axis=1, errors='ignore')  # Ignore if target columns don't exist in test2
test_predictions = best_xgb_model.predict(X_test)

# Create a DataFrame for the predictions
predictions_df = pd.DataFrame(test_predictions, columns=['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C'])

# Optionally, save the predictions to a CSV file
predictions_df.to_csv('test_predictions_xgb_tuned.csv', index=False)

# Display the predictions DataFrame
print(predictions_df)

# Visualize predictions
def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(title)
    plt.show()

# Predict on validation set using the best model
y_pred_val = best_xgb_model.predict(X_val)

# Plot predictions for each target variable
plot_predictions(y_val['Reactivity'], y_pred_val[:, 0], 'Reactivity: Actual vs Predicted')
plot_predictions(y_val['deg_Mg_pH10'], y_pred_val[:, 1], 'deg_Mg_pH10: Actual vs Predicted')
plot_predictions(y_val['deg_Mg_50C'], y_pred_val[:, 2], 'deg_Mg_50C: Actual vs Predicted')

