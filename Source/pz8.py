import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(file_name):
    current_dir = os.path.dirname(__file__)
    project_roof = os.path.dirname(current_dir)
    file_path = os.path.join(project_roof, 'Data', file_name)
    data = pd.read_excel(str(file_path))
    return data


def prepare_data(users, events, orders):

    users['registration_date'] = pd.to_datetime(users['registration_date'])
    events['event_date'] = pd.to_datetime(events['event_date'])
    orders['order_date'] = pd.to_datetime(orders['order_date'])

    events = events.merge(users[['user_id', 'registration_date']], on='user_id', how='left')
    events['age_days'] = (events['event_date'] - events['registration_date']).dt.days

    events_clean = events[(events['age_days'] >= 0) & (events['age_days'] <= 365)].copy()

    return users, events_clean, orders


def analyze_funnel(users, events, orders):

    registered_users = users['user_id'].nunique()

    app_openers = events[events['event_type'].str.lower().str.contains('открытие')]['user_id'].nunique()

    plot_adders = events[events['event_type'].str.lower().str.contains('участк')]['user_id'].nunique()

    valid_orders = orders[orders['user_id'].isin(users['user_id'])]
    buyers = valid_orders['user_id'].nunique()

    stages = ['Регистрация', 'Открытие приложения', 'Добавление участка', 'Первый заказ']
    counts = [registered_users, app_openers, plot_adders, buyers]

    conv_from_prev = [100.0]
    conv_from_base = [100.0]
    for i in range(1, len(counts)):
        prev_conv = (counts[i] / counts[i - 1] * 100) if counts[i - 1] > 0 else 0
        base_conv = (counts[i] / counts[0] * 100) if counts[0] > 0 else 0
        conv_from_prev.append(prev_conv)
        conv_from_base.append(base_conv)

    plot_events = events[events['event_type'].str.lower().str.contains('участк')]
    avg_time_to_plot = plot_events.groupby('user_id')['age_days'].min().mean()

    first_orders = valid_orders.groupby('user_id')['order_date'].min().reset_index()
    first_orders = first_orders.merge(users[['user_id', 'registration_date']], on='user_id', how='left')
    first_orders['time_to_order'] = (first_orders['order_date'] - first_orders['registration_date']).dt.days
    avg_time_to_order = first_orders['time_to_order'].mean()

    print("\n--- Воронка конверсии ---")
    for i in range(len(stages)):
        print(
            f"{stages[i]}: {counts[i]} чел. | От пред. шага: {conv_from_prev[i]:.1f}% | От базы: {conv_from_base[i]:.1f}%")
    print(f"Среднее время от регистрации до добавления участка: {avg_time_to_plot:.1f} дней")
    print(f"Среднее время от регистрации до первого заказа: {avg_time_to_order:.1f} дней")

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(stages, counts, color=['#4c72b0', '#dd8452', '#55a868', '#c44e52'])
    ax.set_title('Воронка конверсии пользователей')
    ax.set_ylabel('Количество пользователей')

    for bar, pct in zip(bars, conv_from_base):
        height = bar.get_height()
        ax.annotate(f'{height}\n({pct:.1f}%)',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')
    plt.tight_layout()
    plt.show()


def cohort_analysis(users, events):

    users['cohort'] = users['registration_date'].dt.to_period('M')
    events_cohort = events.merge(users[['user_id', 'cohort']], on='user_id', how='left')

    def get_period(age):
        if 0 <= age <= 30:
            return '0-30 дней'
        elif 31 <= age <= 60:
            return '31-60 дней'
        elif 61 <= age <= 90:
            return '61-90 дней'
        else:
            return 'Более 90 дней'

    events_cohort['active_period'] = events_cohort['age_days'].apply(get_period)

    target_periods = ['0-30 дней', '31-60 дней', '61-90 дней']
    filtered_events = events_cohort[events_cohort['active_period'].isin(target_periods)]

    cohort_active = filtered_events.groupby(['cohort', 'active_period'])['user_id'].nunique().unstack()

    cohort_sizes = users.groupby('cohort')['user_id'].nunique()

    retention = cohort_active.divide(cohort_sizes, axis=0) * 100
    retention = retention[target_periods]  # Упорядочиваем столбцы

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(retention, annot=True, fmt=".1f", cmap="YlGnBu", ax=ax, cbar_kws={'label': '% Активных пользователей'})
    ax.set_title('Тепловая карта удержания (Retention Rate) по когортам')
    ax.set_ylabel('Когорта (Месяц регистрации)')
    ax.set_xlabel('Период после регистрации')
    plt.tight_layout()
    plt.show()


def financial_metrics(users, events, orders):
    avg_check = orders.groupby('category')['amount_rub'].mean().sort_values(ascending=False)
    print("\n--- Средний чек заказа по категориям ---")
    print(avg_check.apply(lambda x: f"{x:,.2f} руб."))

    orders['order_month'] = orders['order_date'].dt.to_period('M')
    events['event_month'] = events['event_date'].dt.to_period('M')

    monthly_revenue = orders.groupby('order_month')['amount_rub'].sum()

    monthly_active_users = events.groupby('event_month')['user_id'].nunique()

    arpu = (monthly_revenue / monthly_active_users).dropna()

    first_orders = orders.groupby('user_id')['order_date'].min().reset_index()
    first_orders['first_order_month'] = first_orders['order_date'].dt.to_period('M')
    new_paying_users = first_orders.groupby('first_order_month')['user_id'].nunique()

    marketing_budget = monthly_revenue * 0.30
    cac = (marketing_budget / new_paying_users).dropna()

    metrics_df = pd.DataFrame({'ARPU': arpu, 'CAC': cac}).dropna()
    metrics_df.index = metrics_df.index.astype(str)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color = 'tab:blue'
    ax1.set_xlabel('Месяц')
    ax1.set_ylabel('ARPU (руб.)', color=color)
    ax1.plot(metrics_df.index, metrics_df['ARPU'], color=color, marker='o', label='ARPU')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.tick_params(axis='x', rotation=45)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('CAC (руб.)', color=color)
    ax2.plot(metrics_df.index, metrics_df['CAC'], color=color, marker='s', linestyle='--', label='CAC')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Динамика ARPU и CAC по месяцам')
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    df_users = load_data('pz8_users.xlsx')
    df_events = load_data('pz8_events.xlsx')
    df_orders = load_data('pz8_orders.xlsx')

    users_clean, events_clean, orders_clean = prepare_data(df_users, df_events, df_orders)

    analyze_funnel(users_clean, events_clean, orders_clean)

    """
    ===========================================================================
    Вывод: воронка конверсии и время перехода
    ===========================================================================
    Основываясь на визуализации воронки, видно, на каком этапе происходит 
    основной отток пользователей. Часто самым "узким" местом является переход 
    от просмотра прогноза/добавления участка к первому заказу. 

    Зная среднее время от регистрации до заказа, можно настроить цепочки 
    пуш-уведомлений или email-рассылок (например, если среднее время 15 дней, 
    то на 10-й день стоит предложить скидку на первый заказ удобрений). 
    Сезонность также подтверждает эти сроки: весной фермеры быстрее 
    конвертируются в покупателей из-за срочности полевых работ.
    ===========================================================================
    """

    cohort_analysis(users_clean, events_clean)

    """
    ===========================================================================
    Вывод: удержание
    ===========================================================================
    Тепловая карта наглядно показывает, какие когорты "живут" в приложении 
    дольше. Скорее всего, когорты весенних месяцев регистрации (март-май) 
    показывают лучшее удержание в первые 30-60 дней в связи с посевной 
    кампанией. Худшее удержание может наблюдаться у осенне-зимних когорт, 
    что требует внедрения межсезонного функционала (например, обучения, 
    планирования бюджета на следующий год), чтобы пользователи не удаляли 
    приложение в "мертвый" сезон.
    ===========================================================================
    """

    financial_metrics(users_clean, events_clean, orders_clean)

    """
    ===========================================================================
    бизнес-интерпретация и рекомендации 
    ===========================================================================
    Анализ графиков ARPU и CAC позволяет ответить на главный вопрос юнит-экономики: 
    окупается ли привлечение? 
    - Если ARPU стабильно выше CAC: экономика сходится, каждый новый 
      пользователь приносит прибыль, можно масштабировать маркетинг.
    - Если линии пересекаются или CAC > ARPU: компания "сжигает" бюджет.

    Рекомендации для продуктовой команды по повышению конверсии:
    1. Ускорение Onboarding'а: Упростить процесс "Добавления участка" 
       (возможно, интегрировать кадастровую карту или автоопределение по GPS), 
       так как это ключевое действие перед заказом.
    2. Кросс-сейл и Ап-сейл: Внедрить систему товарных рекомендаций 
       (увеличит средний чек и, как следствие, ARPU). Например, при покупке 
       семян предлагать подходящие средства защиты растений.
    3. Работа с когортами: Запустить программу лояльности для удержания 
       зимних когорт на период больше 60 дней, предлагая предзаказы на 
       весну по выгодной цене.
    ===========================================================================
    """