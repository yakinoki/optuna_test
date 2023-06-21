import optuna

def f1(x, y):
    return x + y

def f2(x, y):
    return x * x - y * y

def function(trial):
    # Suggest a float value for the parameter 'x' between -3 and 3.
    x = trial.suggest_float("x", -3, 3)
    # Suggest a float value for the parameter 'y' between -2 and 1.
    y = trial.suggest_float("y", -2, 1)

    # Calculate the value for objective function 1.
    v1 = f1(x, y)
    # Calculate the value for objective function 2.
    v2 = f2(x, y)
    
    return v1, v2

# Create an Optuna study with the directions for both objective functions set to minimize.
study = optuna.create_study(directions=["minimize", "minimize"])
# Optimize the function using 100 trials.
study.optimize(function, n_trials=100)

print("[Best Trials]")
for t in study.best_trials:
    print(f"- [{t.number}] params={t.params}, values={t.values}")

# Plot the Pareto front of the study, including dominated trials.
optuna.visualization.plot_pareto_front(
    study,
    include_dominated_trials=True
).show()
