import pandas as pd

def add_property_age(df):
    current_year = pd.Timestamp.now().year
    df['PropertyAge'] = current_year - pd.to_numeric(df['YearBuilt'], errors='coerce')

    print("\nШапка DataFrame с PropertyAge:")
    print(df[['Suburb', 'YearBuilt', 'PropertyAge']].head())
    return df
