import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import trim_mean
import os
from sklearn.utils import resample


def assessments_of_central_position(df):
    mean = df.mean()
    median = df.median()
    trim_mean_ = trim_mean(df, 0.1)
    return mean, median, trim_mean_


def variability_estimation(df):
    std = df.std()
    iqr = df.quantile(q=0.75) - df.quantile(q=0.25)
    return std, iqr


def plot_boxplot(df):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.boxplot(df)
    ax.set_xticklabels([])
    plt.xlabel('Надой')
    plt.title('Коробчатая диаграмма')
    plt.tight_layout()
    plt.show()


def plot_histogram(df):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist(df, bins=10, edgecolor='black')
    plt.xlabel('Надой молока')
    plt.ylabel('Частота')
    plt.title('Гистограмма')
    plt.tight_layout()
    plt.show()


def plot_scatterplot_hexbin(df):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hexbin(x=df['milk_yield_liters'], y=df['cat_patrol_hours'], alpha=0.5, gridsize=30)
    plt.xlabel('Объем надоя')
    plt.ylabel('Часы кошачьего патрулирования')
    plt.title('Сетка из шестиугольников')
    plt.tight_layout()
    plt.show()


def bootstrap_hist(df):
    results = []
    for _ in range(1000):
        sample = resample(df)
        results.append(sample.mean())

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist(results, bins=10, edgecolor='black')
    plt.xlabel('Удой')
    plt.ylabel('Частота')
    plt.title('Гистограмма по средним из бутстрапа')
    plt.tight_layout()
    plt.show()
    confidence_interval = list(pd.Series(results).quantile([0.025, 0.975]))
    return confidence_interval


if __name__ == "__main__":
    def load_farm_data(file_name):
        # 1. Находим, где мы находимся сейчас
        current_dir = os.path.dirname(__file__)

        # 2. Поднимаемся на уровень выше (в корень проекта)
        project_root = os.path.dirname(current_dir)

        # 3. Собираем путь к файлу в папке data
        # os.path.join сам поставит нужные слеши для Windows или Mac
        file_path = os.path.join(project_root, 'Data', 'farm_data.csv')

        data = pd.read_csv(file_path)
        return data


    farm_data = load_farm_data('farm_data.csv')
    milk_yield = farm_data['milk_yield_liters']
    mean_milk, median_milk, trim_mean_milk = assessments_of_central_position(milk_yield)
    print('Среднее значение: ', round(mean_milk, 3),
          "\nМедиана: ", round(median_milk, 3),
          "\nУсеченное среднее: ", round(trim_mean_milk, 3))

    std_milk, iqr_milk = variability_estimation(milk_yield)
    print("Стандартное отклонение: ", round(std_milk, 3),
          "\nМежквартильный размах: ", round(iqr_milk, 3))
    print('Корреляция между количеством надоя и часами кошачьего патруля: ',
          round((farm_data[['cat_patrol_hours', 'milk_yield_liters']].corr()).iloc[0, 1], 3))
    # Присутствие котиков практически не влияет на надой
    plot_boxplot(milk_yield)
    plot_histogram(milk_yield)
    print('Доверительный интервал: ', bootstrap_hist(milk_yield))
    # Гистограмма напоминает нормальное распределение, в целом все средние значения рано или поздно создадут нормальное распределение,
    # даже если исходные данные такими не являлись
    #привеет
