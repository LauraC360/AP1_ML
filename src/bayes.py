import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.naive_bayes import GaussianNB

def preprocess_data_bayes(train_file_path, test_file_path):
    train_data = pd.read_csv(train_file_path)
    if train_data.empty:
        raise ValueError("Training dataset could not be loaded or is empty.")
    if 'Data' not in train_data.columns:
        raise ValueError("Column 'Data' is missing from the training dataset.")
    try:
        train_data['Data'] = pd.to_datetime(train_data['Data'], format='%Y-%m-%d', dayfirst=True)
    except Exception as e:
        raise ValueError(f"Error in parsing 'Data' column in training dataset: {e}")
    train_data = train_data.dropna()
    train_data = train_data[train_data['Data'].dt.month != 12]
    if train_data.empty:
        raise ValueError("Training data is empty. Ensure the dataset contains data for months other than December.")

    test_data = pd.read_csv(test_file_path)
    if test_data.empty:
        raise ValueError("Test dataset could not be loaded or is empty.")
    if 'Data' not in test_data.columns:
        raise ValueError("Column 'Data' is missing from the test dataset.")
    try:
        test_data['Data'] = pd.to_datetime(test_data['Data'], format='%Y-%m-%d', dayfirst=True)
    except Exception as e:
        raise ValueError(f"Error in parsing 'Data' column in test dataset: {e}")
    test_data = test_data.dropna()
    test_data = test_data[(test_data['Data'].dt.month == 12) & (test_data['Data'].dt.year == 2024)]
    if test_data.empty:
        raise ValueError("Test data is empty. Ensure the dataset contains data for December 2024.")

    features = ['Consum[MW]', 'Productie[MW]', 'Carbune[MW]', 'Hidrocarburi[MW]', 'Ape[MW]',
                'Nuclear[MW]', 'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]']
    if not all(feature in train_data.columns for feature in features):
        missing_features = [feature for feature in features if feature not in train_data.columns]
        raise ValueError(f"Missing features in the training dataset: {missing_features}")
    if not all(feature in test_data.columns for feature in features):
        missing_features = [feature for feature in features if feature not in test_data.columns]
        raise ValueError(f"Missing features in the test dataset: {missing_features}")

    X_train = train_data[features]
    y_train = train_data['Sold[MW]'] if 'Sold[MW]' in train_data.columns else None
    if y_train is None:
        raise ValueError("Column 'Sold[MW]' is missing from the training dataset.")
    X_test = test_data[features]
    y_test = test_data['Sold[MW]'] if 'Sold[MW]' in test_data.columns else None
    if y_test is None:
        raise ValueError("Column 'Sold[MW]' is missing from the test dataset.")

    # Convert y_train to numeric values
    y_train = pd.to_numeric(y_train, errors='coerce').dropna()
    # Convert X_test to numeric values
    X_test = X_test.apply(pd.to_numeric, errors='coerce')

    # Ensure consistent lengths by dropping rows with missing values in both X_test and y_test
    X_test = X_test.dropna()
    y_test = y_test.loc[X_test.index]

    # Discretize continuous variables for Bayes
    X_train_discretized = X_train.apply(lambda x: pd.cut(x, bins=5, labels=False))
    X_test_discretized = X_test.apply(lambda x: pd.cut(x, bins=5, labels=False))

    return X_train_discretized, y_train, X_test_discretized, y_test

def train_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, comparison_results_path):
    try:
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")

        # Save the Bayes results to the comparison results file
        try:
            df = pd.read_csv(comparison_results_path, index_col=0)
        except FileNotFoundError:
            df = pd.DataFrame(index=['RMSE', 'MAE'])

        df['Bayes'] = [rmse, mae]
        df.to_csv(comparison_results_path)
    except ValueError as e:
        print(f"ValueError: {e}")

    # Save predictions and real values to a CSV file
    results = pd.DataFrame({'Real': y_test, 'Predicted': predictions})
    results.to_csv('data/predictions_december_2024.csv', index=False)

if __name__ == "__main__":
    train_file_path = 'data/daily_data.csv'
    test_file_path = 'data/daily_december_2024.csv'
    comparison_results_path = 'data/comparison_results.csv'
    X_train, y_train, X_test, y_test = preprocess_data_bayes(train_file_path, test_file_path)
    model = train_bayes(X_train, y_train)
    evaluate_model(model, X_test, y_test, comparison_results_path)