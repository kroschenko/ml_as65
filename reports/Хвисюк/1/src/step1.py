import pandas as pd

def clean_columns(filepath):
    df = pd.read_csv(filepath, na_values=['missing', 'inf'], low_memory=False)
    missing_counts = df.isna().sum()
    print("Пропуски по столбцам:")
    print(missing_counts.sort_values(ascending=False).head(5))

    most_missing = missing_counts.idxmax()
    print(f"\nУдаляем столбец: {most_missing} ({missing_counts[most_missing]} пропусков)")
    df.drop(columns=[most_missing], inplace=True)

    print(f"Оставшиеся столбцы: {list(df.columns)}")
    return df
