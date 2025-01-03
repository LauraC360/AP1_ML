import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
import graphviz

# Functia pentru preprocesarea datelor
def preprocess_data(train_file_path, test_file_path):

    # Verificare dacă fișierele există
    if not os.path.exists(train_file_path):
        raise FileNotFoundError(f"File not found: {train_file_path}")
    if not os.path.exists(test_file_path):
        raise FileNotFoundError(f"File not found: {test_file_path}")

    # Citirea datelor de antrenament și testare
    train_data = pd.read_csv(train_file_path)

    # Conversie dată la formatul datetime
    train_data['Data'] = pd.to_datetime(train_data['Data'], format='%Y-%m-%d', dayfirst=True)

    # Eliminare date lipsă
    train_data = train_data.dropna()

    # Excluderea datelor pentru luna decembrie
    train_data = train_data[train_data['Data'].dt.month != 12]

    # Preprocesarea datelor de test
    test_data = pd.read_csv(test_file_path)
    test_data['Data'] = pd.to_datetime(test_data['Data'], format='%Y-%m-%d', dayfirst=True)
    test_data = test_data.dropna()
    test_data = test_data[(test_data['Data'].dt.month == 12) & (test_data['Data'].dt.year == 2024)]

    # Selectare caracteristici și ținta
    features = ['Consum[MW]', 'Productie[MW]', 'Carbune[MW]', 'Hidrocarburi[MW]', 'Ape[MW]',
                'Nuclear[MW]', 'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]']
    X_train = train_data[features]
    y_train = train_data['Sold[MW]']
    X_test = test_data[features]
    y_test = test_data['Sold[MW]']

    # Convertirea datelor în format numeric și completarea valorilor lipsă cu 0
    X_train = X_train.apply(pd.to_numeric, errors='coerce').fillna(0)
    y_train = pd.to_numeric(y_train, errors='coerce').fillna(0)
    X_test = X_test.apply(pd.to_numeric, errors='coerce').fillna(0)
    y_test = pd.to_numeric(y_test, errors='coerce').fillna(0)

    return X_train, y_train, X_test, y_test

# Funcție pentru transformarea Sold în variabilă categorică
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

# Functie pentru antrenarea unui clasificator ID3
def train_id3_classifier(X_train, y_train):
    # Definirea hiperparametrilor pentru GridSearch
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }

    # Definirea modelului DecisionTreeClassifier cu criteriul entropy
    model = DecisionTreeClassifier(criterion='entropy')
    # Căutare hiperparametrii optimi folosind GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_ # Selectarea modelului optim
    return best_model

def train_id3_regressor(X_train, y_train):
    # Definirea hiperparametrilor pentru GridSearch
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }

    # Definirea modelului DecisionTreeRegressor cu criteriul squared_error
    model = DecisionTreeRegressor(criterion='squared_error')

    # Căutare hiperparametrii optimi folosind GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_  # Selectarea modelului optim
    return best_model

# Functie pentru vizualizarea arborelui de decizie
def visualize_tree(model, feature_names):
    # Crearea datelor pentru arborele de decizie
    dot_data = export_graphviz(model, out_file=None,
                               feature_names=feature_names,
                               class_names=model.classes_ if hasattr(model, 'classes_') else None,
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    # Exportarea arborelui în format PNG
    graph.format = 'png'
    graph.render("tree")
    graph.view()
    return graph

if __name__ == "__main__":
    # Definirea căilor către fișierele de date
    train_file_path = 'data/daily_data.csv'
    test_file_path = 'data/daily_december_2024.csv'
    comparison_results_path = 'data/comparison_results.csv'

    # Preprocesarea datelor
    X_train, y_train, X_test, y_test = preprocess_data(train_file_path, test_file_path)

    # Antrenarea și evaluarea regresorului ID3
    regressor = train_id3_regressor(X_train, y_train)
    regressor_predictions = regressor.predict(X_test)

    # Calcularea metricilor de evaluare
    rmse = np.sqrt(mean_squared_error(y_test, regressor_predictions))
    mse = mean_squared_error(y_test, regressor_predictions)
    print(f"Regressor - RMSE: {rmse:.2f}, MSE: {mse:.2f}")

    # Categorizarea valorilor pentru clasificare
    y_train_cat = y_train.apply(categorize_sold)
    y_test_cat = y_test.apply(categorize_sold)

    # Antrenarea și evaluarea clasificatorului ID3
    classifier = train_id3_classifier(X_train, y_train_cat)
    classifier_predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test_cat, classifier_predictions)
    print(f"Classifier - Accuracy: {accuracy:.2f}")

    # Salvarea rezultatelor de comparare în fișier
    try:
        df = pd.read_csv(comparison_results_path, index_col=0)
    except FileNotFoundError:
        df = pd.DataFrame(index=['RMSE', 'MSE', 'Accuracy'])

    if len(df.index) != 3:
        df = df.reindex(['RMSE', 'MSE', 'Accuracy'])

    df['ID3_var2'] = [rmse, mse, accuracy]
    df.to_csv(comparison_results_path)