import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import trim_mean
import numpy as np
import wquantiles
from statsmodels import robust


def load_data(file_name):
    current_dir = os.path.dirname(__file__)
    project_roof = os.path.dirname(current_dir)
    file_path = os.path.join(project_roof, 'Data', file_name)
    data = pd.read_csv(str(file_path))
    return data


def central_position_assessments(df, df1):
    return (df.mean(), trim_mean(df, 0.1), df.median(), np.average(df1, weights=df),
            wquantiles.median(df1, weights=df))


def variability_estimates(df):
    return df.std(), df.quantile(0.75) - df.quantile(0.25), robust.scale.mad(df)


def percentiles(df, x):
    return df.quantile(x)


def frequency_table(df):
    binned = pd.cut(df.iloc[:,0], 10)
    binned.name = 'binned'
    df = pd.concat([df, binned], axis=1).copy()
    df = df.sort_values(by=df.columns[0])
    groups = []
    for group, subset in df.groupby(by=df.columns[2], observed=False):
        groups.append({'BinRange': group,
                       'Count': len(subset),
                       'States': ','.join(subset.Abbreviation)})
    print(pd.DataFrame(groups))


def hexbin_plot(df):
    ax = df.plot.hexbin(x=df.columns[0], y=df.columns[1], gridsize=30, sharex=False, figsize=(5,5))
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.tight_layout()
    plt.show()


def kdeplot_plot(df):
    sns.kdeplot(data=df.sample(1000), x=df.columns[0], y=df.columns[1])
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    state = load_data('state.csv')
    kc_tax = load_data('kc_tax.csv')
    kc_tax0 = kc_tax.loc[(kc_tax.TaxAssessedValue < 750000) &
                     (kc_tax.SqFtTotLiving > 100) &
                     (kc_tax.SqFtTotLiving < 3500), :]
    mean, trim_mean, median, weighted_mean, weighted_median = central_position_assessments(state['Population'],
                                                                                           state['Murder.Rate'])
    print(f'Среднее значение: {mean}\n'
          f'Усеченное среднее: {trim_mean}\n'
          f'Медиана: {median}\n'
          f'Среднее взвешенное: {weighted_mean}\n'
          f'Медиана взвешенная: {weighted_median}\n')

    std, interquartile_range, median_absolute_deviation = variability_estimates(state['Population'])
    print(f'Стандартное отклонение: {std}\n'
          f'Межквартильный размах: {interquartile_range}\n'
          f'Медианное абсолютное отклонение от медианы: {median_absolute_deviation}\n')

    percentile = percentiles(state['Population'], 0.05)
    print(f'Процентиль: {percentile}\n')
    frequency_table(state[['Population', 'Abbreviation']])
    hexbin_plot(kc_tax0[['SqFtTotLiving', 'TaxAssessedValue']])
    kdeplot_plot(kc_tax0[['SqFtTotLiving', 'TaxAssessedValue']])
