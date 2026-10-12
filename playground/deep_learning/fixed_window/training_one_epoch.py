import torch
from tqdm.auto import tqdm
import numpy as np


def train_one_epoch(
        model_under_training: torch.nn.Module,
        loss_func: torch.nn.modules.loss,
        optimizer_train: torch.optim,
        loader_train: torch.utils.data.dataloader.DataLoader,
        device_training: torch.device,
        perf_metrics: dict,
):
    model_under_training = model_under_training.to(device=device_training)
    model_under_training.train()

    loss_step: list = []
    accuracy_training = perf_metrics["acc_train"]

    batch_progress_bar = tqdm(enumerate(loader_train), total=len(loader_train), unit="batch", desc="Training Batch")

    for i, data in batch_progress_bar:  # noqa
        # inp_data, labels , inp_data_lengths, labels_lengths = data
        inp_data, labels = data
        inp_data: torch.Tensor = inp_data.float()

        # labels = labels.unsqueeze(1) # BCEwithLogitsLoss
        labels: torch.Tensor = labels.long()  # Normal Crossentropy
        # labels: torch.Tensor = labels.squeeze(dim=0)  # MiniBatch Crossentropy

        # device related moving of data
        # labels = labels.view(labels.shape[0]).to(device_training)
        labels = labels.to(device=device_training)

        inp_data = inp_data.to(device_training)
        inp_data = inp_data.squeeze(dim=0)  # MiniBatch Crossentropy
        outputs: torch.Tensor = model_under_training(inp_data)

        loss: torch.FloatTensor = loss_func(outputs, labels)


        # loss: torch.FloatTensor = loss_func(outputs, loss_label.float())

        msg: str = f"batch loss  = {loss}"
        batch_progress_bar.set_postfix_str(msg)

        optimizer_train.zero_grad()
        loss.backward()
        optimizer_train.step()

        with (torch.no_grad()):
            # _, predicted = torch.max(outputs, 1)
            # predicted class is determined via index position
            _, predicted_class = torch.max(outputs, dim=1)
            loss_step.append(loss.item())
            loss_step.append(loss.detach().cpu().numpy())
            # accuracy_training.update(predicted_class.unsqueeze(dim=0), total_sequence_labels)
            accuracy_training.update(predicted_class, labels)

    # Epoch metrics calculation
    loss_curr_epoch: np.ndarray = np.mean(loss_step)

    accuracy_training_epoch = accuracy_training.compute()

    accuracy_training.reset()

    return (
        loss_curr_epoch,
        accuracy_training_epoch,
    )
