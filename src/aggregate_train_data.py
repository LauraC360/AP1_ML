import pandas as pd

def aggregate_daily_data(data):
    # Transformarea coloanei Data în format datetime
    data['Data'] = pd.to_datetime(data['Data'], dayfirst=True)

    # Convertirea coloanelor relevante la tipuri numerice de date
    columns_to_convert = [
        'Consum[MW]', 'Productie[MW]', 'Carbune[MW]', 'Hidrocarburi[MW]',
        'Ape[MW]', 'Nuclear[MW]', 'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]'
    ]
    for column in columns_to_convert:
        data[column] = pd.to_numeric(data[column], errors='coerce')

    # Gruparea pe zile și agregarea 
    daily_data = data.groupby(data['Data'].dt.date).agg({
        'Consum[MW]': 'sum',
        'Productie[MW]': 'sum',
        'Carbune[MW]': 'sum',
        'Hidrocarburi[MW]': 'sum',
        'Ape[MW]': 'sum',
        'Nuclear[MW]': 'sum',
        'Eolian[MW]': 'sum',
        'Foto[MW]': 'sum',
        'Biomasa[MW]': 'sum'
    }).reset_index()

    # Recalculare 'Sold[MW]' ca diferența dintre 'Productie[MW]' și 'Consum[MW]'
    daily_data['Sold[MW]'] = daily_data['Productie[MW]'] - daily_data['Consum[MW]']

    return daily_data

# Încărcarea dataset-urilor
data_2022 = pd.read_excel('data/Grafic_SEN_2022.xlsx')
data_2023 = pd.read_excel('data/Grafic_SEN_2023.xlsx')

# Combinarea seturilor de date
combined_data = pd.concat([data_2022, data_2023], ignore_index=True)

# Agregarea datelor
daily_data = aggregate_daily_data(combined_data)

# Salvare date agregate într-un nou fișier CSV
daily_data.to_csv('data/daily_data.csv', index=False)

print("Aggregated data saved successfully.")
