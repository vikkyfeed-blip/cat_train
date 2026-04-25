import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


def load_data(file_name):
    current_dir = os.path.dirname(__file__)
    project_roof = os.path.dirname(current_dir)
    file_path = os.path.join(project_roof, 'Data', file_name)
    data = pd.read_csv(str(file_path))
    return data


def violin_plot(df):
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.violinplot(data=df, x=df['section'], y=df['avg_apple_weight'], ax=ax, inner='quartile')
    plt.tight_layout()
    plt.show()


def mean_of_random_samples(df, n):
    result = []
    for i in range(1000):
        result.append(np.mean(df['age_years'].sample(n=n)))
    return result


def hist_plot(s5, s30, s100):
    fig, ax = plt.subplots(figsize=(5, 5), nrows=3, sharex=True)
    ax[0] = s5.plot.hist(bins=20, ax=ax[0])
    ax[1] = s30.plot.hist(bins=20, ax=ax[1])
    ax[2] = s100.plot.hist(bins=20, ax=ax[2])
    plt.show()


def chi_square(df):
    table = df.pivot_table(index='health_status', columns='section', values='tree_id', aggfunc='count')
    print(table)
    chisq, pvalue, df, expected = stats.chi2_contingency(table)
    return pvalue


if __name__ == '__main__':
    apple_data = load_data('apple_data.csv')
    apple_data = apple_data[apple_data['avg_apple_weight'] >= 0]
    print('Корреляция между возрастом дерева и весом яблока:',
          (apple_data[['age_years', 'avg_apple_weight']].corr()).iloc[0, 1])
    violin_plot(apple_data)
    sample_5 = mean_of_random_samples(apple_data, 5)
    sample_30 = mean_of_random_samples(apple_data, 30)
    sample_100 = mean_of_random_samples(apple_data, 100)
    hist_plot(pd.Series(sample_5), pd.Series(sample_30), pd.Series(sample_100))
    print(chi_square(apple_data))
