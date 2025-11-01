import mlflow

experiment = mlflow.get_experiment_by_name("Equipo63_Fase2")
runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

best_run = runs.loc[runs["metrics.final_auc"].idxmax()]
print("Best run ID:", best_run.run_id)
print("Final AUC:", best_run["metrics.final_auc"])
print("Accuracy:", best_run["metrics.final_accuracy"])