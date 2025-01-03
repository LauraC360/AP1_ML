import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.naive_bayes import GaussianNB

# Preprocesarea datelor pentru modelul Bayes
def preprocess_data_bayes(train_file_path, test_file_path):
    # Citirea datelor de antrenare
    train_data = pd.read_csv(train_file_path)
    if train_data.empty:
        raise ValueError("Training dataset could not be loaded or is empty.")
    if 'Data' not in train_data.columns:
        raise ValueError("Column 'Data' is missing from the training dataset.")

    # Convertire coloană Data în formate datetime
    try:
        train_data['Data'] = pd.to_datetime(train_data['Data'], format='%Y-%m-%d', dayfirst=True)
    except Exception as e:
        raise ValueError(f"Error in parsing 'Data' column in training dataset: {e}")

    # Eliminare date lipsă și păstrăm doar lunile diferite de decembrie
    train_data = train_data.dropna()
    train_data = train_data[train_data['Data'].dt.month != 12]
    if train_data.empty:
        raise ValueError("Training data is empty. Ensure the dataset contains data for months other than December.")

    # Similar, preprocesăm setul de testare
    test_data = pd.read_csv(test_file_path)
    if test_data.empty:
        raise ValueError("Test dataset could not be loaded or is empty.")
    if 'Data' not in test_data.columns:
        raise ValueError("Column 'Data' is missing from the test dataset.")
    try:
        test_data['Data'] = pd.to_datetime(test_data['Data'], format='%Y-%m-%d', dayfirst=True)
    except Exception as e:
        raise ValueError(f"Error in parsing 'Data' column in test dataset: {e}")
    # Păstrare date strict din decembrie 2024
    test_data = test_data.dropna()
    test_data = test_data[(test_data['Data'].dt.month == 12) & (test_data['Data'].dt.year == 2024)]
    if test_data.empty:
        raise ValueError("Test data is empty. Ensure the dataset contains data for December 2024.")

    # Definire caracteristici relevante pentru model
    features = ['Consum[MW]', 'Productie[MW]', 'Carbune[MW]', 'Hidrocarburi[MW]', 'Ape[MW]',
                'Nuclear[MW]', 'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]']

    # Verificare dacă toate caracteristicile există în setul de date
    if not all(feature in train_data.columns for feature in features):
        missing_features = [feature for feature in features if feature not in train_data.columns]
        raise ValueError(f"Missing features in the training dataset: {missing_features}")
    if not all(feature in test_data.columns for feature in features):
        missing_features = [feature for feature in features if feature not in test_data.columns]
        raise ValueError(f"Missing features in the test dataset: {missing_features}")

    # Extragere caracteristici și variabila țintă
    X_train = train_data[features]
    y_train = train_data['Sold[MW]'] if 'Sold[MW]' in train_data.columns else None
    if y_train is None:
        raise ValueError("Column 'Sold[MW]' is missing from the training dataset.")
    X_test = test_data[features]
    y_test = test_data['Sold[MW]'] if 'Sold[MW]' in test_data.columns else None
    if y_test is None:
        raise ValueError("Column 'Sold[MW]' is missing from the test dataset.")

    # Convertirea variabilei țintă și datelor caracteristicilor la valori numerice
    y_train = pd.to_numeric(y_train, errors='coerce').dropna()
    X_test = X_test.apply(pd.to_numeric, errors='coerce')

    # Eliminarea rândurilor cu valori lipsă pentru setul de testare
    X_test = X_test.dropna()
    y_test = y_test.loc[X_test.index]

    # Discretizarea variabilelor continue pentru modelul Bayes
    X_train_discretized = X_train.apply(lambda x: pd.cut(x, bins=5, labels=False))
    X_test_discretized = X_test.apply(lambda x: pd.cut(x, bins=5, labels=False))

    return X_train_discretized, y_train, X_test_discretized, y_test

# Funcția pentru categorizarea variabilei de sold
def categorize_sold(value):
    if value < -500:
        return 'Very Low'
    elif -500 <= value < 0:
        return 'Low'
    elif 0 <= value < 500:
        return 'Medium'
    elif 500 <= value < 1000:
        return 'High'
    else:
        return 'Very High'

# Antrenarea modelului Gaussian Naive Bayes
def train_bayes(X_train, y_train):
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

# Evaluarea performanței modelului NB
def evaluate_model(model, X_test, y_test, comparison_results_path):
    try:
        predictions = model.predict(X_test)

        # Calcularea erorilor (RMSE și MSE)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")

        # Categorizarea predicțiilor și valorilor reale pentru calculul acurateții
        y_test_cat = y_test.apply(categorize_sold)
        predictions_cat = pd.Series(predictions).apply(categorize_sold)
        accuracy = accuracy_score(y_test_cat, predictions_cat)
        print(f"Accuracy: {accuracy:.2f}")

        # Salvarea rezultatelor în fișierul pentru comparare
        try:
            df = pd.read_csv(comparison_results_path, index_col=0)
        except FileNotFoundError:
            df = pd.DataFrame(index=['RMSE', 'MAE', 'Accuracy'])

        df['Bayes'] = [rmse, mae, accuracy]
        df.to_csv(comparison_results_path)
    except ValueError as e:
        print(f"ValueError: {e}")

    # Salvarea predicțiilor într-un fișier CSV
    results = pd.DataFrame({'Real': y_test, 'Predicted_Bayes': predictions})
    results.to_csv('data/predictions_december_2024.csv', index=False)

if __name__ == "__main__":
    # Definirea căilor pentru fișiere
    train_file_path = 'data/daily_data.csv'
    test_file_path = 'data/daily_december_2024.csv'
    comparison_results_path = 'data/comparison_results.csv'

    # Preprocesare date
    X_train, y_train, X_test, y_test = preprocess_data_bayes(train_file_path, test_file_path)

    # Antrenarea modelului Naive Bayes
    model = train_bayes(X_train, y_train)

    # Evaluarea modelul
    evaluate_model(model, X_test, y_test, comparison_results_path)