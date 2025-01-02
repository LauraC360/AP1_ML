import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor, export_graphviz
import graphviz

def preprocess_data(train_file_path, test_file_path):
    train_data = pd.read_csv(train_file_path)
    train_data['Data'] = pd.to_datetime(train_data['Data'], format='%Y-%m-%d', dayfirst=True)
    train_data = train_data.dropna()
    train_data = train_data[train_data['Data'].dt.month != 12]

    test_data = pd.read_csv(test_file_path)
    test_data['Data'] = pd.to_datetime(test_data['Data'], format='%Y-%m-%d', dayfirst=True)
    test_data = test_data.dropna()
    test_data = test_data[(test_data['Data'].dt.month == 12) & (test_data['Data'].dt.year == 2024)]

    features = ['Consum[MW]', 'Productie[MW]', 'Carbune[MW]', 'Hidrocarburi[MW]', 'Ape[MW]',
                'Nuclear[MW]', 'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]']
    X_train = train_data[features]
    y_train = train_data['Sold[MW]']
    X_test = test_data[features]
    y_test = test_data['Sold[MW]']

    X_train = X_train.apply(pd.to_numeric, errors='coerce').dropna()
    y_train = pd.to_numeric(y_train, errors='coerce').dropna()
    X_test = X_test.apply(pd.to_numeric, errors='coerce').dropna()
    y_test = pd.to_numeric(y_test, errors='coerce').dropna()

    return X_train, y_train, X_test, y_test

def train_id3(X_train, y_train):
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }
    model = DecisionTreeRegressor(criterion='squared_error')
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    return best_model

def visualize_tree(model, feature_names):
    dot_data = export_graphviz(model, out_file=None,
                               feature_names=feature_names,
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.format = 'png'
    graph.render("tree")
    return graph

if __name__ == "__main__":
    train_file_path = 'data/daily_data.csv'
    test_file_path = 'data/daily_december_2024.csv'
    comparison_results_path = 'data/comparison_results.csv'

    X_train, y_train, X_test, y_test = preprocess_data(train_file_path, test_file_path)
    model = train_id3(X_train, y_train)
    graph = visualize_tree(model, X_train.columns)
    graph.view()

    # Evaluate model
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    # Save the ID3 results to the comparison results file
    try:
        df = pd.read_csv(comparison_results_path, index_col=0)
    except FileNotFoundError:
        df = pd.DataFrame(index=['RMSE', 'MAE'])

    df['ID3'] = [rmse, mae]
    df.to_csv(comparison_results_path)