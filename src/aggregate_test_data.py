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

# Load the dataset
data_2024 = pd.read_excel('data/december_2024.xlsx')

# Aggregate the data
daily_data_2024 = aggregate_daily_data(data_2024)

# Save the aggregated data to a new CSV file
daily_data_2024.to_csv('data/daily_december_2024.csv', index=False)

print("Aggregated data for December 2024 saved successfully.")