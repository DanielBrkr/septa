import torch
import numpy as np
from tqdm.auto import tqdm


def validate(
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
            inp_data, labels, inp_data_lengths, label_lengths = data
            inp_data: torch.Tensor = inp_data.float()
            # labels = labels.unsqueeze(1) # BCEwithLogitsLoss
            labels: torch.Tensor = labels.long()  # Normal Crossentropy

            inp_data = inp_data.to(device_val)
            inp_data = inp_data.squeeze(dim=0)  # MiniBatch Crossentropy
            # labels = labels.view(labels.shape[0]).to(device_val)
            labels = labels.to(device=device_val)

            outputs: torch.Tensor = model_to_validate(inp_data, inp_data_lengths)

            total_sequence_labels, _ = labels.max(dim=1)

            loss_val: torch.FloatTensor = loss_func(outputs, total_sequence_labels)
            predicted: torch.Tensor = torch.argmax(outputs, dim=1)

            loss_step.append(loss_val.item())

            accuracy_validation.update(predicted, total_sequence_labels)

            msg: str = f"batch loss  = {loss_val}"
            batch_progress_bar.set_postfix_str(msg)

    # Epoch metrics calculation
    val_loss_epoch: np.ndarray = torch.tensor(loss_step).mean().numpy()

    accuracy_validation_epoch = accuracy_validation.compute()
    accuracy_validation.reset()

    return (
        val_loss_epoch,
        accuracy_validation_epoch,
    )
