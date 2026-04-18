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
df_exploration, df_distribution, df_ab_test = generate_farm_datasets()

print("Данные готовы! Мур-мяу! 🐱✨")

