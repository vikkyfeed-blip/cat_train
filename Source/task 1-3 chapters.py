from matplotlib import pyplot as plt
import scipy.stats as stats

from data_gen import generate_farm_datasets

df1, df2, df3 = generate_farm_datasets()


def calculate_robust_metrics(df):
    trim_mean_ = stats.trim_mean(df['daily_yield'], 0.1)
    median = df['daily_yield'].median()
    median_abs_deviation = stats.median_abs_deviation(df['daily_yield'])
    return trim_mean_, median, median_abs_deviation


#Робастные метрики это метрики, которые устойчивы к выбросам.


def plot_boxplot(df):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.boxplot(df.daily_yield)
    plt.xlabel("Доходность")
    plt.tight_layout()
    plt.show()


# Основные аргументы для boxplot библиотеки matplotlib:
# — х - сами данные
# — vert - определяет положение графика (вертикально/горизонтально true/false)
# — notch - делает на ящике выемки - доверительный интервал для медианы (есть/нет true/false)
# — patch_artist - дает возможность раскрасить ящик (цветной/не цветной true/false))
# — labels - подписи для данных
# Аргументы для работы с выбросами
# — showfliers - позволяет скрыть все точки выбросов (есть/нет true/false)
# — sym - меняет символ выброса ('r*' сделает выбросы красными звездочками)
# — whis - определяет длину усов (3 - усы станут длиннее)

trim_mean_, median, median_abs_deviation = calculate_robust_metrics(df1)
print("trim_mean: ", trim_mean_)
print("median: ", median)
print("median_abs_deviation: ", median_abs_deviation)
plot_boxplot(df1)
