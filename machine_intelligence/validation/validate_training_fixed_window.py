import torch
import numpy as np
from tqdm.auto import tqdm


def validate_fw(
        model_to_validate: torch.nn.Module,
        loss_func: torch.nn.modules.loss,
        val_loader: torch.utils.data.dataloader.DataLoader,
        device_val: torch.device,
        perf_metrics: dict

):
    model_to_validate.eval()
    loss_step: list = []
    accuracy_validation = perf_metrics["acc_val"]

    batch_progress_bar = tqdm(enumerate(val_loader), total=len(val_loader), unit="batch", desc="Validation Batch")
    with torch.no_grad():
        for i, data in batch_progress_bar:  # noqa
            inp_data,inp_data_static, labels = data

            inp_data: torch.Tensor = inp_data.float()
            inp_data_static: torch.Tensor = inp_data_static.float()
            labels: torch.Tensor = labels.long()  # Normal Crossentropy

            inp_data = inp_data.to(device_val)
            inp_data_static = inp_data_static.to(device_val)
            labels = labels.to(device=device_val)

            outputs_val: torch.Tensor = model_to_validate(inp_data, inp_data_static)
            loss_val: torch.FloatTensor = loss_func(outputs_val, labels)
            loss_step.append(loss_val.detach().cpu().numpy())

            _, predicted_class = torch.max(outputs_val, dim=1)
            accuracy_validation.update(predicted_class, labels)

            msg: str = f"batch loss  = {loss_val}"
            batch_progress_bar.set_postfix_str(msg)

    # Epoch metrics calculation
   #val_loss_epoch: np.ndarray = torch.tensor(loss_step).mean().numpy()
    val_loss_epoch: np.ndarray = np.mean(loss_step)
    accuracy_validation_epoch = accuracy_validation.compute()
    accuracy_validation.reset()

    return (
        val_loss_epoch,
        accuracy_validation_epoch,
    )
