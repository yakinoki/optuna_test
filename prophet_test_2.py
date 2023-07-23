import pandas as pd
import optuna
from prophet import Prophet
import pickle

# データの読み込み
df_test = pd.read_csv('raw_data.csv')
df_predict = pd.read_csv('predict.csv')
df_test.columns = ['ds', 'y']

def calc_max_absolute_error(y_true, y_pred):
    return (y_true - y_pred).abs().max()

def objective(trial):
    # ハイパーパラメータの探索範囲
    changepoint_prior_scale = trial.suggest_loguniform('changepoint_prior_scale', 0.01, 10.0)
    seasonality_prior_scale = trial.suggest_loguniform('seasonality_prior_scale', 0.01, 10.0)
    holidays_prior_scale = trial.suggest_loguniform('holidays_prior_scale', 0.01, 10.0)

    # モデルの定義
    model = Prophet(changepoint_prior_scale=changepoint_prior_scale,
                    seasonality_prior_scale=seasonality_prior_scale,
                    holidays_prior_scale=holidays_prior_scale)

    # 学習
    model.fit(df_test)

    # 予測
    future = model.make_future_dataframe(periods=0)
    forecast = model.predict(future)
    result = forecast[['ds', 'yhat']].tail(len(df_test))

    # 評価指標
    error = calc_max_absolute_error(df_test['y'], result['yhat'])

    # 結果を返す
    return error

# ハイパーパラメータのチューニング
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=10)

with open('best_params.pkl', 'wb') as f:
    pickle.dump(study.best_params, f)

with open('best_params.pkl', 'rb') as f:
    best_params = pickle.load(f)

# ベストなパラメータを使って予測を実行
best_params = study.best_trial.params
model = Prophet(changepoint_prior_scale=best_params['changepoint_prior_scale'],
                seasonality_prior_scale=best_params['seasonality_prior_scale'],
                holidays_prior_scale=best_params['holidays_prior_scale'])
model.fit(df_test)
future = model.make_future_dataframe(periods=len(df_predict))
forecast = model.predict(future)
result = forecast[['ds', 'yhat']]
result_0 = forecast[['ds', 'yhat']].tail(len(df_test))

# 評価指標を表示
error = calc_max_absolute_error(df_test['y'], result_0['yhat'])
print('Max Absolute Error:', error)

# 予測結果を出力
print(result)

result.to_csv('result_predict.csv', index=False, encoding="SHIFT-JISx0213")
