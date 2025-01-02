import pandas as pd

def aggregate_daily_data(data):
    # Ensure the 'Data' column is in datetime format
    data['Data'] = pd.to_datetime(data['Data'], dayfirst=True)

    # Convert relevant columns to numeric, coercing errors to NaN
    columns_to_convert = [
        'Consum[MW]', 'Productie[MW]', 'Carbune[MW]', 'Hidrocarburi[MW]',
        'Ape[MW]', 'Nuclear[MW]', 'Eolian[MW]', 'Foto[MW]', 'Biomasa[MW]'
    ]
    for column in columns_to_convert:
        data[column] = pd.to_numeric(data[column], errors='coerce')

    # Group by date and aggregate the features
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

    # Recalculate 'Sold[MW]' as the difference between 'Productie[MW]' and 'Consum[MW]'
    daily_data['Sold[MW]'] = daily_data['Productie[MW]'] - daily_data['Consum[MW]']

    return daily_data

# Load the datasets
data_2022 = pd.read_excel('data/Grafic_SEN_2022.xlsx')
data_2023 = pd.read_excel('data/Grafic_SEN_2023.xlsx')

# Combine the datasets
combined_data = pd.concat([data_2022, data_2023], ignore_index=True)

# Aggregate the data
daily_data = aggregate_daily_data(combined_data)

# Save the aggregated data to a new CSV file
daily_data.to_csv('data/daily_data.csv', index=False)

print("Aggregated data saved successfully.")