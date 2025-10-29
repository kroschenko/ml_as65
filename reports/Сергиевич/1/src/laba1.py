import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import numpy as np

print("=" * 60)
print("1. ЗАГРУЗКА ДАННЫХ И ИНФОРМАЦИЯ О НИХ")
print("=" * 60)

try:
    df = pd.read_csv("heart.csv")
    print("Данные успешно загружены")
except FileNotFoundError:
    print("Файл 'heart.csv' не найден!")
    print("Создаем демонстрационные данные...")
    np.random.seed(42)
    n_samples = 300

    demo_data = {
        'age': np.random.randint(29, 77, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'trestbps': np.random.randint(94, 200, n_samples),
        'chol': np.random.randint(126, 564, n_samples),
        'thalach': np.random.randint(71, 202, n_samples),
        'target': np.random.randint(0, 2, n_samples)
    }
    df = pd.DataFrame(demo_data)
    print("✓ Демонстрационные данные созданы")


print(f"Размер данных: {df.shape}")
print("\nПервые 5 строк данных:")
print(df.head())

print("\nНазвания колонок:")
print(df.columns.tolist())

print("\nИнформация о типах данных:")
print(df.info())

print("\nПроверка на наличие пропусков:")
missing_data = df.isnull().sum()
print(missing_data)

if missing_data.sum() == 0:
    print("\n✓ Пропуски в данных отсутствуют")
else:
    print(f"\n⚠ Обнаружено пропусков: {missing_data.sum()}")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    print("Пропуски заполнены средними значениями")

print("\nОсновные статистические характеристики:")
print(df.describe())

print("\n" + "=" * 60)
print("2. СТОЛБЧАТАЯ ДИАГРАММА: ЗДОРОВЫЕ VS БОЛЬНЫЕ ПАЦИЕНТЫ")
print("=" * 60)

plt.figure(figsize=(8, 6))
target_counts = df['target'].value_counts().sort_index()


bars = plt.bar(['Здоровые (0)', 'Больные (1)'], target_counts.values,
               color=['lightgreen', 'lightcoral'], edgecolor='black', alpha=0.8)

plt.title('Сравнение количества здоровых и больных пациентов', fontsize=14, fontweight='bold')
plt.xlabel('Категория пациента', fontsize=12)
plt.ylabel('Количество пациентов', fontsize=12)


for bar, count in zip(bars, target_counts.values):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
             str(count), ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Количество здоровых пациентов (target=0): {target_counts[0]}")
print(f"Количество больных пациентов (target=1): {target_counts[1]}")
if len(target_counts) > 1:
    print(f"Соотношение больных/здоровых: {target_counts[1] / target_counts[0]:.2f}")


print("\n" + "=" * 60)
print("3. ДИАГРАММА РАССЕЯНИЯ: ПУЛЬС VS ВОЗРАСТ")
print("=" * 60)

plt.figure(figsize=(10, 7))


if 'thalach' in df.columns and 'age' in df.columns:
    # Создаем диаграмму рассеяния с разными цветами для здоровых и больных
    scatter = plt.scatter(df['age'], df['thalach'],
                          c=df['target'],
                          cmap='coolwarm',
                          alpha=0.7,
                          s=60)

    plt.colorbar(scatter, label='Наличие болезни сердца (0=здоров, 1=болен)')
    plt.title('Зависимость максимального пульса от возраста', fontsize=14, fontweight='bold')
    plt.xlabel('Возраст (лет)', fontsize=12)
    plt.ylabel('Максимальный пульс (thalach)', fontsize=12)

    # Добавляем легенду
    healthy_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                               markersize=8, label='Здоровые (0)')
    sick_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                            markersize=8, label='Больные (1)')
    plt.legend(handles=[healthy_patch, sick_patch])

    # Добавляем линию тренда
    try:
        z = np.polyfit(df['age'], df['thalach'], 1)
        p = np.poly1d(z)
        plt.plot(df['age'], p(df['age']), "g--", alpha=0.8, label='Линия тренда')
        plt.legend()
    except:
        print("⚠ Не удалось построить линию тренда")

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Анализ корреляции
    correlation = df['age'].corr(df['thalach'])
    print(f"Корреляция между возрастом и пульсом: {correlation:.3f}")
    if correlation < 0:
        print("Вывод: наблюдается отрицательная корреляция - с возрастом максимальный пульс снижается")
    else:
        print("Вывод: наблюдается положительная корреляция - с возрастом максимальный пульс увеличивается")
else:
    print("Отсутствуют необходимые колонки 'thalach' или 'age'")


print("\n" + "=" * 60)
print("4. ПРЕОБРАЗОВАНИЕ ПРИЗНАКА SEX И ONE-HOT ENCODING")
print("=" * 60)

if 'sex' in df.columns:
    print("Исходные данные в колонке 'sex':")
    print("0 = женщина, 1 = мужчина")
    print("\nРаспределение до преобразования:")
    print(df['sex'].value_counts().sort_index())

    # Преобразование в читаемый формат
    df['sex_readable'] = df['sex'].map({0: 'female', 1: 'male'})

    print("\nПосле преобразования в читаемый формат:")
    print(df['sex_readable'].value_counts())

    # Применение One-Hot Encoding
    sex_encoded = pd.get_dummies(df['sex_readable'], prefix='sex')
    df = pd.concat([df, sex_encoded], axis=1)

    print("\nРезультат One-Hot Encoding:")
    print(f"Добавлены колонки: {[col for col in df.columns if col.startswith('sex_')]}")
    print("\nПервые 5 строк с преобразованными данными:")
    print(df[['sex', 'sex_readable', 'sex_female', 'sex_male']].head())

    print("\nРаспределение по полу после кодирования:")
    if 'sex_male' in df.columns and 'sex_female' in df.columns:
        print(f"Мужчины: {df['sex_male'].sum()}")
        print(f"Женщины: {df['sex_female'].sum()}")
else:
    print("Колонка 'sex' не найдена в данных")

print("\n" + "=" * 60)
print("5. СРЕДНИЙ УРОВЕНЬ ХОЛЕСТЕРИНА ДЛЯ БОЛЬНЫХ И ЗДОРОВЫХ")
print("=" * 60)

if 'chol' in df.columns and 'target' in df.columns:
    # Расчет средних значений
    chol_sick = df.loc[df['target'] == 1, 'chol'].mean()
    chol_healthy = df.loc[df['target'] == 0, 'chol'].mean()

    chol_sick_std = df.loc[df['target'] == 1, 'chol'].std()
    chol_healthy_std = df.loc[df['target'] == 0, 'chol'].std()

    print(f"Средний уровень холестерина для БОЛЬНЫХ пациентов: {chol_sick:.2f} ± {chol_sick_std:.2f}")
    print(f"Средний уровень холестерина для ЗДОРОВЫХ пациентов: {chol_healthy:.2f} ± {chol_healthy_std:.2f}")

    # Визуализация сравнения
    plt.figure(figsize=(8, 6))
    categories = ['Здоровые', 'Больные']
    means = [chol_healthy, chol_sick]
    std_devs = [chol_healthy_std, chol_sick_std]

    bars = plt.bar(categories, means, yerr=std_devs,
                   capsize=10, color=['lightblue', 'lightcoral'],
                   edgecolor='black', alpha=0.8)

    plt.title('Сравнение среднего уровня холестерина', fontsize=14, fontweight='bold')
    plt.ylabel('Уровень холестерина (chol)', fontsize=12)

    # Добавляем значения на столбцы
    for bar, mean in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f'{mean:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Статистический анализ
    if chol_sick > chol_healthy:
        difference = chol_sick - chol_healthy
        print(f"\nВывод: У больных пациентов уровень холестерина в среднем НА {difference:.2f} единиц ВЫШЕ")
    else:
        difference = chol_healthy - chol_sick
        print(f"\nВывод: У здоровых пациентов уровень холестерина в среднем НА {difference:.2f} единиц ВЫШЕ")
else:
    print("Отсутствуют необходимые колонки 'chol' или 'target'")

print("\n" + "=" * 60)
print("6. НОРМАЛИЗАЦИЯ ПРИЗНАКОВ")
print("=" * 60)

features_to_normalize = ['age', 'trestbps', 'chol', 'thalach']
available_features = [feature for feature in features_to_normalize if feature in df.columns]

if available_features:
    print(f"Признаки для нормализации: {available_features}")

    print("\nДанные ДО нормализации:")
    print(df[available_features].describe())

    # Сохраняем исходные данные для сравнения
    original_data = df[available_features].copy()

    # Выполняем нормализацию
    scaler = MinMaxScaler()
    df_normalized = df.copy()
    df_normalized[available_features] = scaler.fit_transform(df[available_features])

    print("\nДанные ПОСЛЕ нормализации:")
    print(df_normalized[available_features].describe())

    # Визуализация сравнения до и после нормализации
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()

    for i, feature in enumerate(available_features):
        if i < 4:  # Защита от выхода за границы
            axes[i].hist(original_data[feature], alpha=0.7, label='До норм.', bins=15, color='blue')
            axes[i].hist(df_normalized[feature], alpha=0.7, label='После норм.', bins=15, color='red')
            axes[i].set_title(f'Нормализация: {feature}')
            axes[i].set_xlabel('Значение')
            axes[i].set_ylabel('Частота')
            axes[i].legend()

    plt.tight_layout()
    plt.show()

    print("\n✓ Нормализация выполнена успешно!")
    print("Все признаки приведены к диапазону [0, 1]")
else:
    print("Не найдены признаки для нормализации")

# Финальная сводка
print("\n" + "=" * 60)
print("ФИНАЛЬНАЯ СВОДКА")
print("=" * 60)
print(f"Общее количество пациентов: {len(df)}")
print(f"Количество признаков: {len(df.columns)}")
if 'target' in df.columns:
    print(f"Количество здоровых: {len(df[df['target'] == 0])}")
    print(f"Количество больных: {len(df[df['target'] == 1])}")
if 'sex_male' in df.columns and 'sex_female' in df.columns:
    print(f"Соотношение мужчин/женщин: {df['sex_male'].sum()}/{df['sex_female'].sum()}")

# Сохранение обработанных данных
try:
    df.to_csv('heart_disease_processed.csv', index=False)
    print("\n✓ Обработанные данные сохранены в файл 'heart_disease_processed.csv'")
except:
    print("\n⚠ Не удалось сохранить данные в файл")

print("\n" + "=" * 60)
print("ВСЕ ЗАДАНИЯ ВЫПОЛНЕНЫ!")
print("=" * 60)
