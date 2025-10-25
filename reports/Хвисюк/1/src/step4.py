def show_avg_price_by_suburb(df):
    top_suburbs = df['Suburb'].value_counts().nlargest(5).index
    avg_prices = df[df['Suburb'].isin(top_suburbs)].groupby('Suburb')['Price'].mean()

    print("\nСредняя цена по 5 самым популярным пригородам:")
    print(avg_prices)
