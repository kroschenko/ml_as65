import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


print("=" * 70)
print("1. ЗАГРУЗКА ДАННЫХ И ВЫВОД ПЕРВЫХ 10 СТРОК")
print("=" * 70)

df = pd.read_csv('adult.csv')

print("Первые 10 строк данных:")
print(df.head(10))
print("\n")


print("=" * 70)
print("2. АНАЛИЗ И ОБРАБОТКА СТОЛБЦА WORKCLASS")
print("=" * 70)

print("Текущее распределение значений в столбце workclass:")
print(df['workclass'].value_counts())
print("\n")

most_frequent_workclass = df[df['workclass'] != '?']['workclass'].mode()[0]
print(f"Наиболее часто встречающееся значение в workclass (исключая '?'): '{most_frequent_workclass}'")

initial_question_count = (df['workclass'] == '?').sum()
df['workclass'] = df['workclass'].replace('?', most_frequent_workclass)

print(f"Заменено {initial_question_count} значений '?' на '{most_frequent_workclass}'")
print("\nОбновленное распределение значений в столбце workclass:")
print(df['workclass'].value_counts())
print("\n")


print("=" * 70)
print("3. АНАЛИЗ КОЛИЧЕСТВА МУЖЧИН И ЖЕНЩИН")
print("=" * 70)

gender_counts = df['sex'].value_counts()
male_count = gender_counts.get('Male', 0)
female_count = gender_counts.get('Female', 0)

print(f"Количество мужчин: {male_count}")
print(f"Количество женщин: {female_count}")
print(f"Общее количество людей: {len(df)}")
print(f"Доля мужчин: {male_count/len(df)*100:.2f}%")
print(f"Доля женщин: {female_count/len(df)*100:.2f}%")
print("\n")

plt.figure(figsize=(10, 6))

plt.subplot(1, 2, 1)
bars = plt.bar(gender_counts.index, gender_counts.values, color=['lightblue', 'lightpink'], alpha=0.7, edgecolor='black')
plt.title('Количество мужчин и женщин')
plt.xlabel('Пол')
plt.ylabel('Количество')
plt.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height}', ha='center', va='bottom')

plt.subplot(1, 2, 2)
plt.pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
        colors=['lightblue', 'lightpink'], startangle=90)
plt.title('Распределение по полу')

plt.tight_layout()
plt.show()


print("=" * 70)
print("4. ПРЕОБРАЗОВАНИЕ КАТЕГОРИАЛЬНОГО ПРИЗНАКА RACE В ЧИСЛОВОЙ ФОРМАТ")
print("=" * 70)

print("Исходное распределение значений в столбце race:")
print(df['race'].value_counts())
print("\n")

race_mapping = {
    'White': 0,
    'Black': 1,
    'Asian-Pac-Islander': 2,
    'Amer-Indian-Eskimo': 3,
    'Other': 4
}

df['race_numeric'] = df['race'].map(race_mapping)

print("Соответствие категорий числовым значениям:")
for category, numeric_value in race_mapping.items():
    print(f"  {category} -> {numeric_value}")

print("\nРаспределение числовых значений:")
print(df['race_numeric'].value_counts().sort_index())
print("\n")

print("Проверка преобразования (первые 5 строк):")
print(df[['race', 'race_numeric']].head())
print("\n")

print("=" * 70)
print("5. ГИСТОГРАММА РАСПРЕДЕЛЕНИЯ ВОЗРАСТА ПО УРОВНЮ ДОХОДА")
print("=" * 70)

income_column = 'income' if 'income' in df.columns else 'salary'
print(f"Используется столбец дохода: '{income_column}'")

print(f"Распределение по уровням дохода:")
print(df[income_column].value_counts())
print("\n")

plt.figure(figsize=(12, 6))

high_income = df[df[income_column] == '>50K']['age']
low_income = df[df[income_column] == '<=50K']['age']

plt.hist(high_income, bins=30, alpha=0.7, color='green', label='Доход >50K', edgecolor='black')
plt.hist(low_income, bins=30, alpha=0.7, color='red', label='Доход <=50K', edgecolor='black')

plt.title('Распределение возраста по уровням дохода')
plt.xlabel('Возраст')
plt.ylabel('Количество людей')
plt.legend()
plt.grid(alpha=0.3)

plt.text(0.02, 0.98, f'Доход >50K:\nСредний возраст: {high_income.mean():.1f}\nМедиана: {high_income.median():.1f}',
         transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

plt.text(0.02, 0.75, f'Доход <=50K:\nСредний возраст: {low_income.mean():.1f}\nМедиана: {low_income.median():.1f}',
         transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

plt.tight_layout()
plt.show()

print("Статистика возраста по группам дохода:")
print(f"Доход >50K: средний возраст = {high_income.mean():.2f}, медиана = {high_income.median():.2f}")
print(f"Доход <=50K: средний возраст = {low_income.mean():.2f}, медиана = {low_income.median():.2f}")
print("\n")


print("=" * 70)
print("6. СОЗДАНИЕ БИНАРНОГО ПРИЗНАКА IS_USA")
print("=" * 70)

print("Топ-10 стран происхождения:")
print(df['native.country'].value_counts().head(10))
print("\n")

df['is_usa'] = (df['native.country'] == 'United-States').astype(int)

print("Распределение бинарного признака is_usa:")
print(df['is_usa'].value_counts())
print(f"Доля людей из США: {df['is_usa'].mean()*100:.2f}%")
print(f"Доля людей не из США: {(1-df['is_usa'].mean())*100:.2f}%")
print("\n")

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
usa_counts = df['is_usa'].value_counts()
plt.bar(['Не из США', 'Из США'], usa_counts.values, color=['lightcoral', 'lightblue'], alpha=0.7, edgecolor='black')
plt.title('Распределение по стране происхождения')
plt.ylabel('Количество людей')

for i, count in enumerate(usa_counts.values):
    plt.text(i, count + 100, f'{count}', ha='center', va='bottom')

plt.subplot(1, 2, 2)
plt.pie(usa_counts.values, labels=['Не из США', 'Из США'], autopct='%1.1f%%',
        colors=['lightcoral', 'lightblue'], startangle=90)
plt.title('Доля людей из США')

plt.tight_layout()
plt.show()


print("=" * 70)
print("ИТОГОВАЯ СВОДКА ВЫПОЛНЕННЫХ ЗАДАЧ")
print("=" * 70)

print("1. ✓ Загружены данные и выведены первые 10 строк")
print("2. ✓ Проанализирован столбец workclass, заменены '?' на наиболее частые значения")
print("3. ✓ Определено количество мужчин и женщин, построена визуализация")
print("4. ✓ Преобразован категориальный признак race в числовой формат")
print("5. ✓ Построена гистограмма распределения возраста по уровням дохода")
print("6. ✓ Создан бинарный признак is_usa на основе native-country")

print(f"\nИтоговый размер данных: {df.shape}")
print(f"Добавленные столбцы: race_numeric, is_usa")
print("\n" + "=" * 70)
print("ВСЕ ЗАДАЧИ ВЫПОЛНЕНЫ!")
print("=" * 70)

print("\nИнформация о финальном DataFrame:")
print(df.info())