import pandas as pd
import torch
import mlflow
import numpy as np
import torchmetrics as tm
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def random_predictions(num_samples: int) -> torch.Tensor:
    random_preds: torch.Tensor = torch.round(torch.rand(num_samples))
    return random_preds


def true_values(num_samples: int) -> torch.Tensor:
    random_true_values: torch.Tensor = torch.round(torch.rand(num_samples))
    return random_true_values


accuracy_metric = tm.classification.BinaryAccuracy()

num_epochs = 10
num_samples = 1000
threshold = 0.1

tracking_server_uri = "http://192.168.60.144:5010"
mlflow.set_tracking_uri(tracking_server_uri)
mlflow.set_experiment("/test-guestimator-2")

acc_training = []

with mlflow.start_run():
    for epoch in range(num_epochs):
        preds = random_predictions(num_samples)
        trues = true_values(num_samples)

        acc_epoch = accuracy_metric(preds, trues)
        acc_training.append(acc_epoch.numpy())
        print(f"Epoch {epoch + 1}/{num_epochs}, Accuracy: {acc_epoch:.4f}")

        mlflow.log_metric("accuracy", acc_epoch, step=epoch)

    fig, axs = plt.subplots(nrows=1, ncols=1)
    sns.lineplot(pd.DataFrame(data=acc_training), ax=axs, legend=False)
    axs.set_xlabel("Epochs")
    axs.set_ylabel("Accuracy")
    plt.show()

    mlflow.log_figure(figure=fig, artifact_file="accuracy_vs_epochs.png")
