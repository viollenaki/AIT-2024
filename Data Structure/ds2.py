# Data Preprocessing in Machine Learning
# Предварительная обработка данных для машинного обучения

# Step 1: Импорт необходимых библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ПРЕДВАРИТЕЛЬНАЯ ОБРАБОТКА ДАННЫХ ДЛЯ МАШИННОГО ОБУЧЕНИЯ")
print("="*60)

# Step 2: Импорт датасета
print("\nStep 2: Импорт данных...")
try:
    df = pd.read_csv('leads-10000.csv')
    print(f"✓ Данные успешно загружены! Размер датасета: {df.shape}")
    print(f"  Количество строк: {df.shape[0]}")
    print(f"  Количество столбцов: {df.shape[1]}")
except FileNotFoundError:
    print("❌ Файл 'leads-10000.csv' не найден!")
    exit()

# Показать первые 5 строк
print("\nПервые 5 строк данных:")
print(df.head())

# Показать информацию о столбцах
print("\nИнформация о столбцах:")
print(df.info())

# Основная статистика
print("\nОсновная статистическая информация:")
print(df.describe(include='all'))

# Step 3: Проверка пропущенных значений
print("\n" + "="*40)
print("Step 3: Проверка пропущенных значений")
print("="*40)

missing_values = df.isnull().sum()
missing_percentage = (missing_values / len(df)) * 100

print("\nПропущенные значения по столбцам:")
missing_info = pd.DataFrame({
    'Столбец': df.columns,
    'Пропущено': missing_values.values,
    'Процент (%)': missing_percentage.values
})
missing_info = missing_info[missing_info['Пропущено'] > 0].sort_values('Пропущено', ascending=False)

if len(missing_info) > 0:
    print(missing_info.to_string(index=False))
    
    # Визуализация пропущенных значений
    if len(missing_info) > 0:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        missing_info.plot(x='Столбец', y='Пропущено', kind='bar', ax=plt.gca())
        plt.title('Количество пропущенных значений')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.subplot(1, 2, 2)
        missing_info.plot(x='Столбец', y='Процент (%)', kind='bar', ax=plt.gca(), color='orange')
        plt.title('Процент пропущенных значений')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        plt.show()
else:
    print("✓ Пропущенные значения отсутствуют!")

# Step 4: Анализ категориальных переменных
print("\n" + "="*40)
print("Step 4: Анализ категориальных переменных")
print("="*40)

# Определение категориальных и числовых столбцов
categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nКатегориальные столбцы ({len(categorical_columns)}):")
for col in categorical_columns:
    print(f"  - {col}")

print(f"\nЧисловые столбцы ({len(numerical_columns)}):")
for col in numerical_columns:
    print(f"  - {col}")

# Анализ категориальных переменных
print("\nДетальный анализ категориальных переменных:")
for col in categorical_columns[:5]:  # Показываем первые 5 для экономии места
    print(f"\n{col}:")
    print(f"  Уникальных значений: {df[col].nunique()}")
    if df[col].nunique() <= 20:  # Показываем распределение только если уникальных значений мало
        print("  Распределение значений:")
        value_counts = df[col].value_counts()
        for value, count in value_counts.head(10).items():
            percentage = (count / len(df)) * 100
            print(f"    {value}: {count} ({percentage:.1f}%)")

# Создание копии для обработки
df_processed = df.copy()

# Обработка пропущенных значений (если есть)
if len(missing_info) > 0:
    print("\nОбработка пропущенных значений...")
    
    # Для числовых столбцов - заполнение медианой
    numeric_imputer = SimpleImputer(strategy='median')
    for col in numerical_columns:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col] = numeric_imputer.fit_transform(df_processed[[col]]).flatten()
            print(f"  ✓ {col}: пропуски заполнены медианой")
    
    # Для категориальных столбцов - заполнение модой (наиболее частым значением)
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    for col in categorical_columns:
        if df_processed[col].isnull().sum() > 0:
            df_processed[col] = categorical_imputer.fit_transform(df_processed[[col]]).flatten()
            print(f"  ✓ {col}: пропуски заполнены наиболее частым значением")

# Кодирование категориальных переменных
print("\nКодирование категориальных переменных...")

# Label Encoding для категориальных переменных
label_encoders = {}
df_encoded = df_processed.copy()

for col in categorical_columns:
    if col != 'Index':  # Исключаем индекс
        le = LabelEncoder()
        df_encoded[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
        label_encoders[col] = le
        print(f"  ✓ {col} -> {col}_encoded")

# Step 5: Разделение датасета на тренировочную и тестовую выборки
print("\n" + "="*40)
print("Step 5: Разделение на тренировочную и тестовую выборки")
print("="*40)

# Предполагаем, что 'Deal Stage' - это наша целевая переменная
target_column = 'Deal Stage'
if target_column in df.columns:
    # Подготовка признаков (X) и целевой переменной (y)
    # Используем закодированные версии категориальных переменных
    feature_columns = []
    
    # Добавляем числовые столбцы (кроме Index)
    for col in numerical_columns:
        if col != 'Index':
            feature_columns.append(col)
    
    # Добавляем закодированные категориальные столбцы
    for col in categorical_columns:
        if col not in [target_column, 'Index']:
            feature_columns.append(col + '_encoded')
    
    X = df_encoded[feature_columns]
    y = df_encoded[target_column + '_encoded']
    
    print(f"Количество признаков: {X.shape[1]}")
    print(f"Целевая переменная: {target_column}")
    print(f"Уникальные классы в целевой переменной: {df[target_column].nunique()}")
    print(f"Классы: {df[target_column].unique()}")
    
    # Разделение на тренировочную и тестовую выборки (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nРазмеры выборок:")
    print(f"  Тренировочная выборка: {X_train.shape}")
    print(f"  Тестовая выборка: {X_test.shape}")
    print(f"  Соотношение: {X_train.shape[0]/(X_train.shape[0] + X_test.shape[0])*100:.0f}% / {X_test.shape[0]/(X_train.shape[0] + X_test.shape[0])*100:.0f}%")

else:
    print(f"❌ Столбец '{target_column}' не найден!")
    print("Доступные столбцы:", df.columns.tolist())

# Step 6: Масштабирование признаков
print("\n" + "="*40)
print("Step 6: Масштабирование признаков (Feature Scaling)")
print("="*40)

if 'X_train' in locals():
    # Стандартизация признаков
    scaler = StandardScaler()
    
    # Обучаем scaler только на тренировочных данных
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Преобразуем обратно в DataFrame для удобства
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print("✓ Масштабирование выполнено успешно!")
    print(f"  Средние значения после масштабирования (должны быть ~0):")
    print(f"    Тренировочная выборка: {X_train_scaled.mean(axis=0).mean():.2e}")
    print(f"    Тестовая выборка: {X_test_scaled.mean(axis=0).mean():.2e}")
    
    print(f"  Стандартные отклонения после масштабирования (должны быть ~1):")
    print(f"    Тренировочная выборка: {X_train_scaled.std(axis=0).mean():.3f}")
    print(f"    Тестовая выборка: {X_test_scaled.std(axis=0).mean():.3f}")
    
    # Сравнение до и после масштабирования
    print("\nСравнение статистик до и после масштабирования (первые 5 признаков):")
    comparison_df = pd.DataFrame({
        'Признак': X_train.columns[:5],
        'Среднее (до)': X_train.iloc[:, :5].mean().values,
        'Среднее (после)': X_train_scaled_df.iloc[:, :5].mean().values,
        'Ст.откл. (до)': X_train.iloc[:, :5].std().values,
        'Ст.откл. (после)': X_train_scaled_df.iloc[:, :5].std().values
    })
    print(comparison_df.round(3).to_string(index=False))

# Финальная сводка
print("\n" + "="*60)
print("ИТОГОВАЯ СВОДКА ПО ПРЕДВАРИТЕЛЬНОЙ ОБРАБОТКЕ")
print("="*60)

print("✓ Step 1: Библиотеки импортированы")
print(f"✓ Step 2: Данные загружены ({df.shape[0]} строк, {df.shape[1]} столбцов)")
if len(missing_info) > 0:
    print(f"✓ Step 3: Пропущенные значения обработаны ({len(missing_info)} столбцов)")
else:
    print("✓ Step 3: Пропущенные значения отсутствуют")
print(f"✓ Step 4: Категориальные переменные закодированы ({len(categorical_columns)} столбцов)")
if 'X_train' in locals():
    print(f"✓ Step 5: Данные разделены на выборки (тренировочная: {X_train.shape[0]}, тестовая: {X_test.shape[0]})")
    print("✓ Step 6: Признаки масштабированы")
else:
    print("❌ Step 5-6: Не выполнены из-за проблем с целевой переменной")

print("\nДанные готовы для применения алгоритмов машинного обучения!")

# Сохранение обработанных данных
print("\nСохранение обработанных данных...")
if 'X_train_scaled_df' in locals():
    # Сохраняем обработанные данные
    processed_data = {
        'X_train': X_train_scaled_df,
        'X_test': X_test_scaled_df, 
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': X_train.columns.tolist(),
        'target_classes': df[target_column].unique().tolist(),
        'label_encoders': label_encoders,
        'scaler': scaler
    }
    
    # Можно сохранить в pickle для дальнейшего использования
    import pickle
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump(processed_data, f)
    
    print("✓ Обработанные данные сохранены в 'processed_data.pkl'")

print("\nПредварительная обработка данных завершена!")