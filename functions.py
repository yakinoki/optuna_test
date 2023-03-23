import optuna

def function(trial):
    x = trial.suggest_float("x", -3, 3)
    y = trial.suggest_float("y", -2, 1)
    
    return x*x + 2 * y *y 

study = optuna.create_study(direction="minimize")
study.optimize(function, n_trials=100)

print(f"Best function value: {study.best_value}")
print(f"Best parameter: {study.best_params}")