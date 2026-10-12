import torch
from tqdm.auto import tqdm
import numpy as np

from torch.profiler import profile, record_function, ProfilerActivity

def train_one_epoch_fw(
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

    batch_progress_bar = tqdm(
        enumerate(loader_train), total=len(loader_train), unit="batch", desc="Training Batch"
    )

    for i, data in batch_progress_bar:  # noqa
        inp_data, inp_data_static, labels = data

        inp_data: torch.Tensor = inp_data.float()

        inp_data_static: torch.Tensor = inp_data_static.float()
        labels: torch.Tensor = labels.long()  # Normal Crossentropy

        inp_data = inp_data.to(device_training)
        inp_data_static = inp_data_static.to(device_training)
        labels = labels.to(device=device_training)

        outputs_train: torch.Tensor = model_under_training(inp_data, inp_data_static)

        loss_train: torch.FloatTensor = loss_func(outputs_train, labels)

        optimizer_train.zero_grad()
        loss_train.backward()
        optimizer_train.step()


        with torch.no_grad():
            # predicted class is determined via index position - an output contains
            # a tuple with the softmaxed out for each class
            loss_step.append(loss_train.detach().cpu().numpy())

            _, predicted_class = torch.max(outputs_train, dim=1)
            accuracy_training.update(predicted_class, labels)

        msg: str = f"batch loss  = {loss_train}"
        batch_progress_bar.set_postfix_str(msg)




    # Epoch metrics calculation
    loss_curr_epoch: np.ndarray = np.mean(loss_step)
    accuracy_training_epoch = accuracy_training.compute()
    accuracy_training.reset()

    return (
        loss_curr_epoch,
        accuracy_training_epoch,
    )
