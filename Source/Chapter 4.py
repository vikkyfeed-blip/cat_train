import os.path
import matplotlib.pyplot as plt
import pandas as pd


def load_df(file_name):
    current_dir = os.path.dirname(__file__)
    project_root = os.path.dirname(current_dir)
    file_path = os.path.join(project_root, "Data", file_name)
    data = pd.read_csv(str(file_path))
    return data


def regression_equation(df):
    lung.plot.scatter(x='Exposure', y='PEFR')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    house_sale = load_df('house_sales.csv')
    lung = load_df('LungDisease.csv')
    regression_equation(lung)