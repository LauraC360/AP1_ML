import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
import graphviz

# Functie pentru preprocesarea datelor
def preprocess_data(train_file_path, test_file_path):
    train_data = pd.read_csv(train_file_path)

    # Converse dată la formatul datetime
    train_data['Data'] = pd.to_datetime(train_data['Data'], format='%Y-%m-%d', dayfirst=True)
    train_data = train_data.dropna() # Eliminare date lipsă

    # Exclude luna december din setul de date de antrenament dacă există cumva
    train_data = train_data[train_data['Data'].dt.month != 12]

    # Procesare date de test
    test_data = pd.read_csv(test_file_path)
    test_data['Data'] = pd.to_datetime(test_data['Data'], format='%Y-%m-%d', dayfirst=True)
    test_data = test_data.dropna()
    test_data = test_data[(test_data['Data'].dt.month == 12) & (test_data['Data'].dt.year == 2024)]

    # Agregarea
    # Adăugare coloane noi pentru datele de antrenament și testare
    # Intermitent[MW] = Eolian[MW] + Foto[MW], Constant[MW] = Nuclear[MW] + Carbune[MW] + Hidrocarburi[MW]
    # Aceste coloane sunt adăugate pentru a simplifica modelul împărțind datele în două categorii după tipul de producție
    train_data['Intermittent[MW]'] = train_data['Eolian[MW]'] + train_data['Foto[MW]']
    train_data['Constant[MW]'] = train_data['Nuclear[MW]'] + train_data['Carbune[MW]'] + train_data['Hidrocarburi[MW]']
    test_data['Intermittent[MW]'] = test_data['Eolian[MW]'] + test_data['Foto[MW]']
    test_data['Constant[MW]'] = test_data['Nuclear[MW]'] + test_data['Carbune[MW]'] + test_data['Hidrocarburi[MW]']

    # Selectarea caracteristicilor relevante
    features = ['Consum[MW]', 'Productie[MW]', 'Intermittent[MW]', 'Constant[MW]', 'Ape[MW]', 'Biomasa[MW]']
    X_train = train_data[features]
    y_train = train_data['Sold[MW]']
    X_test = test_data[features]
    y_test = test_data['Sold[MW]']

    # Conversie la tip numeric și eliminare date lipsă (dacă există)
    X_train = X_train.apply(pd.to_numeric, errors='coerce').dropna()
    y_train = pd.to_numeric(y_train, errors='coerce').dropna()
    X_test = X_test.apply(pd.to_numeric, errors='coerce').dropna()
    y_test = pd.to_numeric(y_test, errors='coerce').dropna()

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

# Funcție pentru antrenarea unui clasificator ID3
def train_id3_classifier(X_train, y_train):
    # Definirea hiperparametrilor pentru GridSearch
    param_grid = {
        'max_depth': [3, 5, 7, 10],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5]
    }

    # Inițializarea modelului ID3 pentru clasificare
    model = DecisionTreeClassifier(criterion='entropy')

    # Căutare hiperparametri optimi cu validare încrucișată (cross-validation)
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

    # Inițializarea modelului ID3 pentru regresie
    model = DecisionTreeRegressor(criterion='squared_error')

    # Căutare hiperparametri optimi cu validare încrucișată (cross-validation)
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_ # Selectarea modelului cu cea mai bună performanță
    return best_model

# Funcția pentru vizualizarea arborelui de decizie
def visualize_tree(model, feature_names):
    # Exportarea arborelui în format DOT
    dot_data = export_graphviz(model, out_file=None,
                               feature_names=feature_names,
                               class_names=model.classes_ if hasattr(model, 'classes_') else None,
                               filled=True, rounded=True,
                               special_characters=True)
    graph = graphviz.Source(dot_data)
    # Exportarea arborelui într-un fișier PNG
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

    # Calcularea metricilor de evaluare pentru regresor
    rmse = np.sqrt(mean_squared_error(y_test, regressor_predictions))
    mae = mean_absolute_error(y_test, regressor_predictions)
    print(f"Regressor - RMSE: {rmse:.2f}, MAE: {mae:.2f}")

    # Categorisirea valorilor din coloana 'Sold[MW]' pentru clasificare
    y_train_cat = y_train.apply(categorize_sold)
    y_test_cat = y_test.apply(categorize_sold)

    # Antrenarea și evaluarea clasificatorului ID3
    classifier = train_id3_classifier(X_train, y_train_cat)
    classifier_predictions = classifier.predict(X_test)
    accuracy = accuracy_score(y_test_cat, classifier_predictions)
    print(f"Classifier - Accuracy: {accuracy:.2f}")

    # Vizualizarea arborelui de decizie
    visualize_tree(classifier, X_train.columns)

    # Salvarea rezultatelor de comparare în fișier
    try:
        df = pd.read_csv(comparison_results_path, index_col=0)
    except FileNotFoundError:
        df = pd.DataFrame(index=['RMSE', 'MAE', 'Accuracy'])

    if len(df.index) != 3:
        df = df.reindex(['RMSE', 'MAE', 'Accuracy'])

    df['ID3'] = [rmse, mae, accuracy]
    df.to_csv(comparison_results_path)