import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay


def load_data(file_name):
    current_dir = os.path.dirname(__file__)
    project_roof = os.path.dirname(current_dir)
    file_path = os.path.join(project_roof, 'Data', file_name)
    data = pd.read_excel(str(file_path))
    return data


def plot_correlation(df):
    numeric_df = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    plt.title('Корреляционная матрица')
    plt.tight_layout()
    plt.show()


def box_plot(df):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=df, x='crop', y='actual_profit_rub_ha', ax=ax)
    plt.title('Распределение прибыли по культурам')
    plt.tight_layout()
    plt.show()


def hist_plot(df):
    fig, ax = plt.subplots(figsize=(10, 4), nrows=1, ncols=2)
    df['yield_t_ha'].plot.hist(bins=20, ax=ax[0], title='Гистограмма урожайности')
    df['actual_profit_rub_ha'].plot.hist(bins=20, ax=ax[1], title='Гистограмма прибыли')
    plt.tight_layout()
    plt.show()


def prepare_data(df):
    df = df.dropna()
    X = df[['region', 'nitrogen_ppm', 'annual_rainfall_mm', 'crop']]
    y = df['success_flag']
    X_encoded = pd.get_dummies(X, columns=['region', 'crop'], drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, X_encoded.columns, scaler


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'LogisticRegression': LogisticRegression(random_state=42),
        'RandomForestClassifier': RandomForestClassifier(random_state=42),
        'GradientBoostingClassifier': GradientBoostingClassifier(random_state=42)
    }

    best_model = None
    best_f1 = 0
    best_name = ""

    print("\n--- Сравнение моделей ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')

        print(f"{name}:")
        print(f"  Точность (Accuracy): {acc:.3f}, F1-score: {f1:.3f}")
        print(f"  Кросс-валидация F1 (среднее): {cv_scores.mean():.3f} (станд. откл.: {cv_scores.std():.3f})")

        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_name = name

    print(f"\nЛучшая модель по метрике F1: {best_name}")
    return best_model


def plot_roc_and_cm(model, X_test, y_test):
    fig, ax = plt.subplots(figsize=(12, 5), nrows=1, ncols=2)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax[0], cmap='Blues')
    ax[0].set_title('Матрица ошибок')

    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    ax[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC-кривая (area = {roc_auc:.2f})')
    ax[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax[1].set_xlabel('False Positive Rate')
    ax[1].set_ylabel('True Positive Rate')
    ax[1].set_title('ROC-кривая')
    ax[1].legend(loc="lower right")

    plt.tight_layout()
    plt.show()


def recommend_crop(model, scaler, feature_columns):
    test_plot_data = {
        'region': 'South',
        'nitrogen_ppm': 55.0,
        'annual_rainfall_mm': 400.0
    }

    crops = ['Wheat', 'Corn', 'Sunflower', 'Potato']
    recommendations = []

    for crop in crops:
        df_test = pd.DataFrame([{**test_plot_data, 'crop': crop}])

        df_encoded = pd.get_dummies(df_test, columns=['region', 'crop'])

        for col in feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[feature_columns]

        X_scaled = scaler.transform(df_encoded)

        prob_success = model.predict_proba(X_scaled)[0][1]
        recommendations.append({'Культура': crop, 'Вероятность успеха': prob_success})

    results_df = pd.DataFrame(recommendations).sort_values(by='Вероятность успеха', ascending=False)
    print("\n--- Рекомендации для участка (Регион: South, Азот: 55, Осадки: 400 мм) ---")
    print(results_df.to_string(index=False))


if __name__ == '__main__':
    climate_data = load_data('pz7_climate.xlsx')
    history_data = load_data('pz7_history.xlsx')
    plots_data = load_data('pz7_plots.xlsx')

    merged_data = pd.merge(history_data, plots_data, on='plot_id', how='inner')
    full_data = pd.merge(merged_data, climate_data, on=['region', 'year'], how='inner')

    plot_correlation(full_data)
    box_plot(full_data)
    hist_plot(full_data)

    """
    ===========================================================================
    Вывод: анализ и гипотезы
    ===========================================================================
    На основе проведенного предварительного анализа можно сформулировать 
    следующие 3 гипотезы:
    1. Успешность выращивания влаголюбивых культур (например, кукурузы) 
       сильно зависит от годовой суммы осадков.
    2. Высокое содержание азота (nitrogen_ppm) в почве является критическим 
       фактором для рентабельности азотозависимых культур (пшеница).
    3. Регион посадки (климатические условия и плодородность) оказывает 
       значимое влияние на итоговую прибыль и класс успешности (success_flag).
    ===========================================================================
    """

    X_train, X_test, y_train, y_test, features, scaler = prepare_data(full_data)
    best_model = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    plot_roc_and_cm(best_model, X_test, y_test)

    """
    ===========================================================================
    Вывод: бизнес-интерпретация
    ===========================================================================
    Бизнес-интерпретация результатов:
    Обученные модели подтвердили, что климатические данные и характеристики почвы
    позволяют с высокой долей вероятности предсказывать экономическую успешность 
    посадки. Кросс-валидация показала стабильность модели. Наилучшие результаты 
    (как правило, у GradientBoostingClassifier или RandomForest) свидетельствуют 
    о нелинейном характере зависимостей (например, избыток влаги может быть 
    так же плох, как и ее недостаток). ROC-кривая демонстрирует высокую 
    способность модели разделять убыточные и прибыльные исходы.
    ===========================================================================
    """

    recommend_crop(best_model, scaler, features)

    """
    ===========================================================================
    Вывод: рекомендации и улучшение модели
    ===========================================================================
    Рекомендации для пользователя:
    - Для исследуемого участка (Южный регион, средний азот, низкие осадки) 
      наиболее перспективно рассмотреть посадку засухоустойчивых культур 
      (например, подсолнечника).
    - Влаголюбивые культуры (кукуруза) при заданном уровне осадков рекомендуются 
      к посадке только при наличии систем дополнительного орошения.
    - Выращивание пшеницы также возможно, однако рекомендуется провести 
      дополнительную оценку экономической целесообразности внесения удобрений.

    Ограничения модели:
    - Модель не учитывает текущие рыночные цены на культуры и затраты на 
      логистику, которые могут сильно варьироваться.
    - Исторические климатические данные могут быть нерепрезентативны 
      в условиях глобального изменения климата (аномальные засухи или заморозки).
    - Отсутствуют данные по вредителям и болезням растений, которые могут 
      свести на нет урожай даже при идеальной почве.
    ===========================================================================
    """