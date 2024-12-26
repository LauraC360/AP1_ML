import pandas as pd
import numpy as np
from openpyxl import load_workbook
from openpyxl.styles import NamedStyle, Font

def apply_default_style(file_path):
    wb = load_workbook(file_path)
    if 'default' not in wb.named_styles:
        default_style = NamedStyle(name='default')
        default_style.font = Font(name='Arial', size=10)
        wb.add_named_style(default_style)
        wb.save(file_path)

# Aplicarea stilului implicit pe fișierele xlsx de date
apply_default_style('data/Grafic_SEN_dec2022.xlsx')
apply_default_style('data/Grafic_SEN_dec2023.xlsx')

# Citirea datelor din fișierele xlsx de date
try:
    data2022 = pd.read_excel('data/Grafic_SEN_dec2022.xlsx', engine='openpyxl')
    data2023 = pd.read_excel('data/Grafic_SEN_dec2023.xlsx', engine='openpyxl')
except Exception as e:
    print(f"Error reading Excel files: {e}")
    raise

# Impartirea datelor in setul de antrenare si cel de testare + transformarile numerice

# Setul de antrenare
data = pd.concat([data2022, data2023])

# Salvare set de antrenare (csv)
data.to_csv('data/combined_data.csv', index=False)

# Conversia setului de antrenare la o listă de dicționare
train_data_list = data.to_dict(orient='records')

# Preprocesarea datelor
data = data.dropna()  # Eliminarea valorilor lipsă
data['Data'] = pd.to_datetime(data['Data'], format='%d-%m-%Y %H:%M:%S')  # Conversia coloanei 'Data' la tipul datetime

# Explorarea datelor
print(data.describe())