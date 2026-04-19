import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import trim_mean

farm_data = pd.read_csv(r"W:\FUCK YOU MICROSOFT\cat_train\Data\farm_data.csv")
milk_yield = farm_data['milk_yield_liters']


def assessments_of_central_position(df):
    mean = df.mean()
    median = df.median()
    trim_mean_ = trim_mean(df, 0.1)
    return mean, median, trim_mean_


mean_milk, median_milk, trim_mean_milk = assessments_of_central_position(milk_yield)
print('Среднее значение: ', round(mean_milk, 3),
      "\nМедиана: ", round(median_milk, 3),
      "\nУсеченное среднее: ", round(trim_mean_milk, 3))


def variability_estimation(df):
    std = df.std()
    iqr = df.quantile(q=0.75) - df.quantile(q=0.25)
    return std, iqr
std_milk, iqr_milk = variability_estimation(milk_yield)
print("Стандартное отклонение: ", round(std_milk, 3),
      "\nМежквартильный размах: ", round(iqr_milk, 3))

def plot_boxplot(df):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.boxplot(df)
    ax.set_xticklabels([])
    plt.xlabel('Надой')
    plt.title('Коробчатая диаграмма')
    plt.tight_layout()
    plt.show()


#plot_boxplot(milk_yield)

def plot_histogram(df):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.hist(df, bins=10)
    plt.xlabel('Надой молока')
    plt.ylabel('Частота')
    plt.title('Гистограмма')
    plt.tight_layout()
    plt.show()


#plot_histogram(milk_yield)


def plot_scatterplot_hexbin(df):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.hexbin(x=df['milk_yield_liters'], y=df['cat_patrol_hours'], alpha=0.5, gridsize=30)
    plt.xlabel('Объем надоя')
    plt.ylabel('Часы кошачьего патрулирования')
    plt.title('Сетка из шестиугольников')
    plt.tight_layout()
    plt.show()
    correlation = df[['cat_patrol_hours', 'milk_yield_liters']].corr()
    return correlation.iloc[0, 1]


correl_milk = plot_scatterplot_hexbin(farm_data)
print('Корреляция между количеством надоя и часами кошачьего патруля: ', round(correl_milk, 3))
#Присутствие котиков практически не влияет на надой