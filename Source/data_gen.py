import pandas as pd
import numpy as np

# Установим random seed, чтобы результаты всегда были одинаковыми
np.random.seed(42)


def generate_farm_datasets():
    # --- ДАТАСЕТ 1 (Для Главы 1: Разведочный анализ) ---
    # По мотивам примера с популяцией штатов и выбросами
    # Создаем данные о надоях 50 коровок (в литрах) с явными выбросами
    milk_yields = np.random.normal(loc=25, scale=5, size=45).tolist()
    outliers = [150, 145, 2, 1, 0]  # Сломанные датчики или супер-коровки
    dataset_1 = pd.DataFrame({
        'cow_id': range(1, 51),
        'daily_yield': milk_yields + outliers
    })

    # --- ДАТАСЕТ 2 (Для Главы 2: Распределения и Бутстрап) ---
    # По мотивам примера с выборками и стандартной ошибкой [cite: 7, 9]
    # Данные о весе 200 телят (в кг), распределение с "длинным хвостом"
    dataset_2 = pd.DataFrame({
        'calf_id': range(1, 201),
        'weight': np.random.lognormal(mean=3.9, sigma=0.3, size=200)  # Длинный хвост [cite: 2]
    })

    # --- ДАТАСЕТ 3 (Для Главы 3: A/B тестирование) ---
    # По мотивам примера с "прилипчивостью страниц" и таблицами сопряженности [cite: 6, 17]
    # Сравниваем старый корм (A) и новый клубничный корм (B)
    group_a = np.random.normal(loc=20, scale=2, size=100)  # Контрольная группа
    group_b = np.random.normal(loc=21.5, scale=2, size=100)  # Тестовая группа

    dataset_3 = pd.DataFrame({
        'group': ['A'] * 100 + ['B'] * 100,
        'milk_production': np.concatenate([group_a, group_b])
    })

    return dataset_1, dataset_2, dataset_3


# Загружаем данные в переменные
# df_exploration, df_distribution, df_ab_test = generate_farm_datasets()


def generate_farm_data(n_cows=1000):
    """
    Генерирует реалистичный набор данных для фермы «Hello Kitty Meadows».
    """
    # Устанавливаем seed для воспроизводимости результатов
    np.random.seed(42)

    # Идентификаторы коровок
    cow_ids = [f"COW_{i:04d}" for i in range(1, n_cows + 1)]

    # Породы (категориальные данные)
    # Голштинская, Джерсейская и Айрширская породы в заданных пропорциях
    breeds = np.random.choice(['Holstein', 'Jersey', 'Ayrshire'], size=n_cows, p=[0.6, 0.3, 0.1])

    # Диета для A/B тестирования (группы эксперимента)
    # Стандартная диета против инновационного клевера "Hello Kitty"
    diets = np.random.choice(['Standard', 'Hello_Kitty_Clover'], size=n_cows, p=[0.5, 0.5])

    # Часы патрулирования котиков в неделю (непрерывная переменная)
    # Сколько часов милые котики гуляли рядом с коровником, создавая уют и защищая от мышей
    cat_hours = np.random.normal(loc=20, scale=5, size=n_cows)
    cat_hours = np.clip(cat_hours, 0, 50)  # Ограничиваем значения в разумных пределах

    # Базовый надой молока (литров в неделю) — наше популяционное среднее
    base_yield = np.random.normal(loc=150, scale=20, size=n_cows)

    # Добавляем эффекты (влияние факторов на целевую переменную)
    # Эффект от диеты: клевер дает прибавку в 5.5 литров
    diet_effect = np.where(diets == 'Hello_Kitty_Clover', 5.5, 0)
    # Эффект котиков: счастливые коровы дают чуть больше молока (коэффициент 0.3)
    cat_effect = cat_hours * 0.3

    # Итоговый надой с учетом всех факторов
    milk_yield = base_yield + diet_effect + cat_effect

    # Вносим выбросы (Outliers)
    # Например, у некоторых коровок была неудачная неделя, и надой резко упал
    outlier_indices = np.random.choice(n_cows, size=15, replace=False)
    milk_yield[outlier_indices] = milk_yield[outlier_indices] * np.random.uniform(0.3, 0.5, size=15)

    # Собираем всё в прямоугольные данные (DataFrame) [cite: 23, 24]
    df = pd.DataFrame({
        'cow_id': cow_ids,
        'breed': breeds,
        'diet_type': diets,
        'cat_patrol_hours': cat_hours,
        'milk_yield_liters': milk_yield
    })

    return df

# Чтобы начать работу, запусти эти строки:
# farm_df = generate_farm_data()
# farm_df.to_csv('farm_data.csv', index=False)
