import os.path
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
    model = LinearRegression()
    model.fit(predictors, outcome)
    return model.intercept_, model.coef_[0]


if __name__ == "__main__":
    house_sale = load_df('house_sales.csv')
    lung = load_df('LungDisease.csv')
    # regression_equation(lung)
    predictors_lung = ['Exposure']
    outcome_lung = 'PEFR'
    intercept, coef = linearregression(predictors_lung, outcome_lung)
    print("Пересечение b0: ", intercept,
          "\nКоэффициент b1: ", coef)
    ox = '1'
