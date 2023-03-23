import optuna

def f1(x, y):
    return x + y

def f2(x, y):
    return x * x - y * y

def function(trial):
    x = trial.suggest_float("x", -3, 3)
    y = trial.suggest_float("y", -2, 1)

    v1 = f1(x, y)
    v2 = f2(x, y)
    
    return v1, v2

study = optuna.create_study(directions = ["minimize", "minimize"])
study.optimize(function, n_trials=100)

print("[Best Trials]")
for t in study.best_trials:
    print(f"- [{t.number}] params={t.params}, values={t.values}")
#print(f"Best function value: {study.best_value}")
#print(f"Best parameter: {study.best_params}")



optuna.visualization.plot_pareto_front(
    study,
    include_dominated_trials=True
).show()