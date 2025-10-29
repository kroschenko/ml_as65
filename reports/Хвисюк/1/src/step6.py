import pandas as pd

def encode_type_column(df):
    df = pd.get_dummies(df, columns=['Type'], prefix='Type')

    print("\nНовые столбцы после One-Hot Encoding:")
    print([col for col in df.columns if col.startswith('Type_')])

    print("\nШапка итогового DataFrame:")
    print(df.head())
    return df
