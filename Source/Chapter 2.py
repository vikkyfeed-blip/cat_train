import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#from resample.bootstrap import confidence_interval
from sklearn.utils import resample
from scipy import stats
import numpy as np
#Sampling Distribution of a Statistic
loans_income = pd.read_csv(r'C:\Users\Viloska\Downloads\loans_income.csv').squeeze('columns')

sample_data = pd.DataFrame({
    'income': loans_income.sample(1000),
    'type': 'Data',
})

sample_data_05 = pd.DataFrame({
    'income': [loans_income.sample(5).mean() for _ in range(1000)],
    'type': 'Mean of 5',
})

sample_data_20 = pd.DataFrame({
    'income': [loans_income.sample(20).mean() for _ in range(1000)],
    'type': 'Mean of 20',
})

results = pd.concat([sample_data, sample_data_05, sample_data_20])
print(results.head())

g = sns.FacetGrid(results, col='type', col_wrap=1,
                  height=2, aspect=2)
g.map(plt.hist, 'income', range=[0,200000], bins=40)
g.set_axis_labels('Income', 'Count')
g.set_titles('{col_name}')

plt.tight_layout()
plt.show()
#Bootstrap
results = []
for nrepeat in range(1000):
    sample = resample(loans_income)
    results.append(sample.median())
results = pd.Series(results)
print('Bootstrap Statistics:')
print(f'original: {loans_income.median()}')
print(f'bias: {results.mean() - loans_income.median()}')
print(f'std. error: {results.std()}')

#confidence intervals
print(loans_income.mean())
np.random.seed(3)
sample20 = resample(loans_income, n_samples=20, replace=False)
print(sample20.mean())
results = []
for nrepeat in range(500):
    sample = resample(sample20)
    results.append(sample.mean())
results = pd.Series(results)

confidence_interval = list(results.quantile([0.05, 0.95]))
ax = results.plot.hist(bins=30, figsize = (4,3))
ax.plot(confidence_interval, [55,55], color='black')
for x in confidence_interval:
    ax.plot([x,x], [0,65], color='black')
    ax.text(x, 70, f'{x:.0f}',
            horizontalalignment='center', verticalalignment='center',)

meanIncome = results.mean()
ax.plot([meanIncome,meanIncome], [0,50], color='black', linestyle='--')
ax.text(meanIncome, 10, f'Mean: {meanIncome:.0f}',
        bbox = dict(facecolor='white', edgecolor='white', alpha = 0.5),
        horizontalalignment='center', verticalalignment='center',)
ax.set_ylim(0,80)
ax.set_ylabel('Counts')

plt.tight_layout()
plt.show()

#Standart Normal and QQ-Plots

fig, ax = plt.subplots(figsize=(4, 4))
norm_sample = stats.norm.rvs(size=100)
stats.probplot(norm_sample, plot=ax)
plt.tight_layout()
plt.show()
print(stats.norm.rvs(size=100, std=10, mean=170))