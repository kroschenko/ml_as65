import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("C:\\Users\\wlksm\\OneDrive\\Рабочий стол\\Уник\\OMO\\ml_as65\\reports\\Kolbashko\\src\\Melbourne_housing.csv")

missing_values = df.isnull().sum()
column_with_most_missing = missing_values.idxmax()
df = df.drop(columns=[column_with_most_missing])
print(f"\nСтолбец с наибольшим количеством пропусков: {column_with_most_missing} ({missing_values.max()} пропусков)")

df = df.dropna(subset=['Price'])

plt.figure(figsize=(12, 6))
plt.hist(df['Price'], bins=50, edgecolor='black', alpha=0.7)
plt.title('Распределение цен на недвижимость в Мельбурне')
plt.xlabel('Цена')
plt.ylabel('Количество домов')
plt.grid(alpha=0.3)
plt.show()

top_5_suburbs = df['Suburb'].value_counts().head(5).index
print("\n5 самых популярных пригородов:")
print(top_5_suburbs.tolist())

for suburb in top_5_suburbs:
    avg_price = df[df['Suburb'] == suburb]['Price'].mean()
    print(f"{suburb}: средняя цена = {avg_price:,.2f}")

current_year = pd.Timestamp.now().year
df['PropertyAge'] = current_year - df['YearBuilt']
print(f"\nСтатистика по возрасту недвижимости:")
print(df['PropertyAge'].describe())


