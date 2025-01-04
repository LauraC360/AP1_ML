import pandas as pd

def aggregate_daily_data(data):
    # Transformare coloană Data în format datetime
    data['Data'] = pd.to_datetime(data['Data'], dayfirst=True)

    # Convertirea coloanelor relevante la tipul numeric acolo unde e cazul
    columns_to_convert = [
        'Consum[MW]', 'Productie[MW]', 'Carbune[MW]', 'Hidrocarburi[MW]',
        'Ape[MW]', 'Nuclear[MW]', 'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]'
    ]
    for column in columns_to_convert:
        data[column] = pd.to_numeric(data[column], errors='coerce')

    # Gruparea pe zile și agregarea datelor
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

# Citirea fișierului excel
data_2024 = pd.read_excel('data/december_2024.xlsx')

# Agregare date
daily_data_2024 = aggregate_daily_data(data_2024)

# Salvarea noilor date agregate într-un fișier CSV nou
daily_data_2024.to_csv('data/daily_december_2024.csv', index=False)

print("Aggregated data for December 2024 saved successfully.")
