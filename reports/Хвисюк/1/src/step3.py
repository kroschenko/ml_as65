import matplotlib.pyplot as plt

def plot_price_distribution(df):
    plt.figure(figsize=(10, 6))
    plt.hist(df['Price'], bins=40, color='skyblue', edgecolor='black')
    plt.title('Распределение цен на недвижимость в Мельбурне')
    plt.xlabel('Цена (AUD)')
    plt.ylabel('Количество объектов')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
