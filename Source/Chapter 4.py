import os.path

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd


def load_df(file_name):
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(current_dir)
    file_path = os.path.join(project_root, "Data", file_name)
    data = pd.read_csv(str(file_path))
    return data


def regression_equation(df):
    lung.plot.scatter(x=df['Exposure'], y=df['PEFR'])
    plt.tight_layout()
    plt.show()


"""Эта функция находит коэффициенты простой линейной регрессии"""
def linearregression(predictors, outcome):
    model = LinearRegression() # создание пустой формы для будущей модели
    model.fit(predictors, outcome) # обучение - определение коэффициентов
    return model.intercept_, model.coef_[0], model # intercept_ - b0, coef_ - b1


def plot(model):
    fig, ax = plt.subplots(figsize=(4,4))
    ax.set_xlim(0, 23)
    ax.set_ylim(295, 450)
    ax.set_xlabel('Exposure')
    ax.set_ylabel('PEFR')
    ax.plot((0,23), model.predict(pd.DataFrame({'Exposure': [0,23]})))
    ax.text(0.4, model.intercept_[0], r'$b_0$', size='larger')

    x = pd.DataFrame({'Exposure': [7.5, 17.5]})
    y = model.predict(x).flatten() #flatten превращает многомерный массив в одномерный, predict -                  #предсказывает значенния для опр. х
    ax.plot((7.5, 7.5, 17.5), (y[0], y[1], y[1]), '--')
    ax.text(5, np.mean(y), r'$\Delta Y$', size='larger')
    ax.text(12, y[1] - 10, r'$\Delta X$', size='larger')
    ax.text(12, 390, r'$b_1 = \frac{\Delta Y}{\Delta X}$', size='larger')
    plt.tight_layout()
    plt.show()


def fitted_values_and_residuals(df, fitted):
    ax = df.plot.scatter(x='Exposure', y='PEFR', figsize=(4,4))
    ax.plot(lung.Exposure, fitted)
    for x, yactual, yfitted in zip (lung.Exposure, lung.PEFR, fitted):
        ax.plot((x,x), (yactual, yfitted), '--', color='C1')
    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    house_sale = load_df('house_sales.csv')
    lung = load_df('LungDisease.csv')
    # regression_equation(lung)
    predictors_lung = lung[['Exposure']]
    outcome_lung = lung[['PEFR']]
    intercept, coef, model_lung = linearregression(predictors_lung, outcome_lung)
    print("Пересечение b0: ", intercept,
          "\nКоэффициент b1: ", coef)
    #plot(model_lung)
    fitted_lung = model_lung.predict(predictors_lung)
    residuals = outcome_lung - fitted_lung #остатки, наблюдаемое значение - предсказанное
    fitted_values_and_residuals(lung, fitted_lung.flatten())
