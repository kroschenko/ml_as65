def drop_missing_price(df):
    before = len(df)
    df = df.dropna(subset=['Price'])
    after = len(df)

    print(f"\nУдалено строк без цены: {before - after}")
    print(f"Оставшиеся строки: {after}")
    return df
