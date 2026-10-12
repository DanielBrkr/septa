from typing import Any

import torch
import torch.nn as nn
import lightning as lit
import wandb
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchmetrics
from loguru import logger
from pathlib import Path
import pandas as pd
from sklearn import metrics

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from machine_intelligence.utilities.visualizations_performance import (
    generate_test_performance_report,
)

from machine_intelligence.utilities.viz_setup import figure_setup


class FixWinLSTM(lit.LightningModule):
    def __init__(
            self,
            num_features_ts: int,
            num_features_static: int,
            loss_fn,
            optimizer_name: str,
            optimizer_hparams: dict,
            arch_config: dict
    ):
        super().__init__()
        self.num_features_ts: int = num_features_ts
        self.num_features_static: int = num_features_static
        self.num_hidden_units: int = arch_config["num_hidden_size"]
        self.num_layers: int = arch_config["num_layers"]
        self.num_ts_out: int = arch_config["num_ts_out"]
        # self.num_features_classifier = self.num_hidden_units + self.num_features_static
        self.num_static_hidden = arch_config["num_static_hidden"]
        self.num_static_outs = arch_config["num_static_outs"]
        self.num_classifier_head_hidden = arch_config["num_classifier_head_hidden"]
        self.dropout = arch_config["dropout"]
        self.dropout_static = arch_config["dropout"]

        self.save_hyperparameters()

        self.loss_fn = loss_fn

        self.init_metrics()

        # -------- time series variables branch
        self.lstm = nn.LSTM(
            input_size=self.num_features_ts,
            hidden_size=self.num_hidden_units,
            batch_first=True,  # doesn't apply on hidden or cell states
            num_layers=self.num_layers,
            bidirectional=False,
            proj_size=0,  # is the default
        )

        self.time_series_end_stage = nn.Sequential(
            nn.Linear(in_features=self.num_hidden_units, out_features=self.num_ts_out),
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        self.static_layers = nn.Sequential(
            nn.Linear(
                in_features=self.num_features_static, out_features=self.num_static_hidden
            ),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_static),
            nn.Linear(in_features=self.num_static_hidden, out_features=self.num_static_outs),
        )

        dim_head_in = self.num_static_outs + self.num_ts_out

        self.classifier_head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(p=self.dropout_static),
            nn.Linear(in_features=dim_head_in, out_features=self.num_classifier_head_hidden),
            nn.ReLU(),
            nn.Dropout(p=self.dropout_static),
            nn.Linear(self.num_classifier_head_hidden, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x_ts_in, x_static_in):
        # time series variables branch
        # h_0, c_0 defaults are zeros, no need to do it manually
        # https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html

        # lstm_out: (N,L,D∗Hout) with N: batch_size, L: sequence length,
        # D: D=2 if bidirectional=True otherwise 1, Hout: hidden_size

        lstm_out, (h_n, c_n) = self.lstm(x_ts_in)
        hidden_state_last = h_n[-1]

        x_ts = self.time_series_end_stage(hidden_state_last)

        x_static = self.static_layers(x_static_in)

        x = torch.cat([x_ts, x_static], dim=1)
        x = self.classifier_head(x)

        return x

    def configure_optimizers(self) -> OptimizerLRScheduler:
        if self.hparams.optimizer_name == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = torch.optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            assert False, f"Unknown optimizer: {self.hparams.optimizer_name}"

        scheduler = {
            # "scheduler": ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10, verbose=True),
            "scheduler": ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=3, verbose=True),
            "monitor": "val/loss",
            "interval": "epoch",
            "frequency": 1
        }

        logger.info(
            f"Configured optimizer {self.hparams.optimizer_name} with hyperparams: {self.hparams.optimizer_hparams}"
        )

        return [optimizer], [scheduler]

    def init_metrics(self):

        self.accuracy: [torchmetrics.Metric] = nn.ModuleList(
            [
                torchmetrics.classification.Accuracy(task="binary", average="macro")
                for _ in range(3)
            ]
        )
        self.specificity: [torchmetrics.Metric] = nn.ModuleList(
            [
                torchmetrics.classification.Specificity(task="binary", average="macro")
                for _ in range(3)
            ]
        )
        self.precision: [torchmetrics.Metric] = nn.ModuleList(
            [
                torchmetrics.classification.Precision(task="binary", average="macro")
                for _ in range(3)
            ]
        )
        self.recall: [torchmetrics.Metric] = nn.ModuleList(
            [
                torchmetrics.classification.Recall(task="binary", average="macro")
                for _ in range(3)
            ]
        )
        self.f1_score: [torchmetrics.Metric] = nn.ModuleList(
            [torchmetrics.classification.F1Score(task="binary") for _ in range(3)]
        )

        self.prec_at_recall: [torchmetrics.Metric] = nn.ModuleList(
            [torchmetrics.classification.PrecisionAtFixedRecall(min_recall=0.8,
                                                                task="binary") for _ in
             range(3)]
        )

        self.stat_scores: [torchmetrics.Metric] = nn.ModuleList(
            [
                torchmetrics.classification.StatScores(task="binary", threshold=0.5)
                for _ in range(3)
            ]
        )
        self.prc_rec_crv: [torchmetrics.Metric] = nn.ModuleList(
            [
                torchmetrics.classification.PrecisionRecallCurve(task="binary")
                for _ in range(3)
            ]
        )

        self.roc_crv: [torchmetrics.Metric] = nn.ModuleList(
            [
                torchmetrics.classification.ROC(task="binary")
                for _ in range(3)
            ]
        )

        self.confusion: [torchmetrics.Metric] = nn.ModuleList(
            [torchmetrics.ConfusionMatrix(task="binary", threshold=0.8) for _ in range(3)]
        )

        self.confusion_50: [torchmetrics.Metric] = nn.ModuleList(
            [torchmetrics.ConfusionMatrix(task="binary", threshold=0.5) for _ in range(3)]
        )

        self.auroc: [torchmetrics.Metric] = nn.ModuleList(
            [torchmetrics.AUROC(task="binary") for _ in range(3)]
        )
        # this should serve as a container for the metrics
        self.metrics: [torchmetrics.Metric] = nn.ModuleList(
            [
                self.accuracy,
                self.specificity,
                self.precision,
                self.recall,
                self.prec_at_recall,
                self.f1_score,
                self.auroc,
            ]
        )

    def _shared_eval(self, batch, batch_idx, prefix):

        inp_data, inp_data_static, labels = batch

        inp_data: torch.Tensor = inp_data.float()
        inp_data_static: torch.Tensor = inp_data_static.float()
        # labels: torch.Tensor = labels.long()  # Normal Crossentropy
        labels: torch.Tensor = labels.float()  # BCE

        outputs = self.forward(inp_data, inp_data_static)

        loss: torch.FloatTensor = self.loss_fn(outputs.squeeze(1), labels)

        if prefix != "predict":
            self.log(f"{prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        preds_probabilities = torch.sigmoid(outputs).squeeze(1)

        threshold = 0.5  # threshold may vary
        preds = (preds_probabilities >= threshold).int()

        return preds_probabilities, preds, labels, loss

    def training_step(self, batch, batch_idx):

        preds_proba, preds, labels, loss_train = self._shared_eval(batch, batch_idx, prefix="train")

        # train index is 0
        for metric in self.metrics:
            metric[0].update(preds_proba, labels.long())

        self.confusion[0].update(preds_proba, labels)
        self.prc_rec_crv[0].update(preds_proba.to(torch.double), labels.to(torch.int8))
        self.roc_crv[0].update(preds_proba.to(torch.double), labels.to(torch.int8))

        return loss_train

    def on_train_epoch_end(self) -> None:
        accuracy_on_train: torch.Tensor = self.accuracy[0].compute()
        specificity_on_train: torch.Tensor = self.specificity[0].compute()
        precision_on_train: torch.Tensor = self.precision[0].compute()
        recall_on_train: torch.Tensor = self.recall[0].compute()
        prec_at_recall_on_train: torch.Tensor = self.prec_at_recall[0].compute()
        f1_on_train: torch.Tensor = self.f1_score[0].compute()
        auroc_on_train: torch.Tensor = self.auroc[0].compute()

        prc_rec_train: torch.Tensor = self.prc_rec_crv[0].compute()
        prc_rec_train_list = [item.tolist() for item in prc_rec_train]

        pr_curve_x = prc_rec_train_list[1][::-1]
        pr_curve_y = prc_rec_train_list[0][::-1]

        df_prc = pd.DataFrame(data={"Recall": pr_curve_x, "Precision": pr_curve_y})

        auprc_on_train: float = metrics.auc(pr_curve_x, pr_curve_y)

        self.log_dict(
            {
                "train/acc": accuracy_on_train,
                "train/spec": specificity_on_train,
                "train/prec": precision_on_train,
                "train/recall": recall_on_train,
                "train/recall@80": prec_at_recall_on_train[0],
                "train/f1": f1_on_train,
                "train/auroc": auroc_on_train,
                "train/auprc": auprc_on_train,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        for metric in self.metrics:
            metric[0].reset()

        self.confusion[0].reset()

    def on_train_end(self) -> None:
        roc_crv_train: torch.Tensor = self.roc_crv[0].compute()
        roc_crv_train_list = [item.tolist() for item in roc_crv_train]
        # fpr, tpr, thresholds = roc(pred, target)

        roc_curve_x = roc_crv_train_list[0][::-1]  # false positive rate
        roc_curve_y = roc_crv_train_list[1][::-1]  # sensitivity

        fig_roc, axs, _, _ = figure_setup(num_cols=1, num_rows=1, width_cm=10, height_cm=10)

        auc: float = metrics.auc(roc_curve_x, roc_curve_y)

        sns.lineplot(
            x=roc_curve_x, y=roc_curve_y, ax=axs, sort=False, estimator=None, color="#9345c9"
        )

        axs.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

        auc_legend = plt.Line2D(
            [0],
            [0],
            color="#9345c9",
            marker="o",
            markerfacecolor="#9345c9",
            label=f"AUC: {auc:.3f}",
        )
        axs.legend(
            handles=[auc_legend],
            loc="best",
            fancybox=True,
            facecolor="#9345c9",
            shadow=False,
            framealpha=0.2,
        )

        # axs_pr.set_title("Precision-Recall Curve")
        axs.set_xlabel("False Positive Rate")
        axs.set_ylabel("Sensitivity")
        # axs_pr.grid()
        axs.set_title("On Last Training Epoch")
        plt.tight_layout()

        self.trainer.loggers[1].experiment.log(
            {
                "train/roc_crv": wandb.Image(fig_roc),
            }
        )

        plt.show()

        df_roc = pd.DataFrame(data={"False Positive Rate": roc_curve_x, "Sensitivity": roc_curve_y})

        wandb.log(
            {"train/roc-curve-wand": wandb.plot.line(wandb.Table(dataframe=df_roc), "False Positive Rate",
                                                     "Sensitivity",
                                                     title="ROC Curve")}, )

        self.roc_crv[0].reset()

    def validation_step(self, batch, batch_idx):

        preds_proba, preds, labels, loss_val = self._shared_eval(batch, batch_idx, prefix="val")

        # train index is 1
        for metric in self.metrics:
            metric[1].update(preds_proba, labels.long())

        self.confusion[1].update(preds_proba, labels)
        self.prc_rec_crv[1].update(preds_proba.to(torch.double), labels.to(torch.int8))

        return loss_val

    def on_validation_epoch_end(self) -> None:
        accuracy_on_val: torch.Tensor = self.accuracy[1].compute()
        specificity_on_val: torch.Tensor = self.specificity[1].compute()
        precision_on_val: torch.Tensor = self.precision[1].compute()
        recall_on_val: torch.Tensor = self.recall[1].compute()
        prec_at_recall_on_val: torch.Tensor = self.prec_at_recall[1].compute()
        f1_on_val: torch.Tensor = self.f1_score[1].compute()
        auroc_on_val: torch.Tensor = self.auroc[1].compute()

        prc_rec_val: torch.Tensor = self.prc_rec_crv[1].compute()
        prc_rec_val_list = [item.tolist() for item in prc_rec_val]

        pr_curve_x = prc_rec_val_list[1][::-1]
        pr_curve_y = prc_rec_val_list[0][::-1]

        auprc_on_val: float = metrics.auc(pr_curve_x, pr_curve_y)

        self.log_dict(
            {
                "val/acc": accuracy_on_val,
                "val/spec": specificity_on_val,
                "val/prec": precision_on_val,
                "val/recall": recall_on_val,
                "val/recall@80": prec_at_recall_on_val[0],
                "val/f1": f1_on_val,
                "val/auroc": auroc_on_val,
                "val/auprc": auprc_on_val
            },
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        for metric in self.metrics:
            metric[1].reset()

        self.confusion[1].reset()
        self.prc_rec_crv[1].reset()

    def test_step(self, batch, batch_idx):
        preds_proba, preds_test, labels, loss = self._shared_eval(batch, batch_idx, prefix="test")

        # test index is 2
        for metric in self.metrics:
            metric[2].update(preds_proba, labels.int())

        self.confusion[2].update(preds_proba, labels.int())
        self.confusion_50[2].update(preds_proba, labels.int())
        self.prc_rec_crv[2].update(preds_proba.to(torch.double), labels.to(torch.int8))
        self.roc_crv[2].update(preds_proba.to(torch.double), labels.to(torch.int8))
        self.stat_scores[2].update(preds_proba.to(torch.double), labels.to(torch.int8))

    def on_test_epoch_end(self) -> None:
        accuracy_on_test: torch.Tensor = self.accuracy[2].compute()
        specificity_on_test: torch.Tensor = self.specificity[2].compute()
        precision_on_test: torch.Tensor = self.precision[2].compute()
        recall_on_test: torch.Tensor = self.recall[2].compute()
        prec_at_recall_on_test: torch.Tensor = self.prec_at_recall[2].compute()
        stat_scores_on_test: torch.Tensor = self.stat_scores[2].compute()
        f1_on_test: torch.Tensor = self.f1_score[2].compute()
        auroc_on_test: torch.Tensor = self.auroc[2].compute()

        confusion_on_test: torch.Tensor = self.confusion[2].compute()
        confusion_50_on_test: torch.Tensor = self.confusion_50[2].compute()

        prc_rec_crv_on_test: torch.Tensor = self.prc_rec_crv[2].compute()
        prc_rec_test_list = [item.tolist() for item in prc_rec_crv_on_test]

        # fpr = fp / (fp + tn)
        fpr_on_test = stat_scores_on_test[1] / (stat_scores_on_test[1] + stat_scores_on_test[2])

        pr_curve_x = prc_rec_test_list[1][::-1]
        pr_curve_y = prc_rec_test_list[0][::-1]

        auprc_on_test: float = metrics.auc(pr_curve_x, pr_curve_y)

        metrics_scalar = {
            "test/acc": accuracy_on_test,
            "test/spec": specificity_on_test,
            "test/prec": precision_on_test,
            "test/recall": recall_on_test,
            "test/recall@80": prec_at_recall_on_test[0],
            "test/f1": f1_on_test,
            "test/fpr": fpr_on_test,
            "test/auroc": auroc_on_test,
            "test/auprc": auprc_on_test,
        }

        metrics_tensor = {
            "test/prc_rec_crv": prc_rec_crv_on_test,
            "test/confusion": confusion_on_test,
            "test/confusion50": confusion_50_on_test,
        }

        # Todo: Save metrics in csv (if this works)

        self.log_dict(
            metrics_scalar,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )

        if self.loggers[1].save_dir is None:
            metrics_log_dir = Path("experiment_logs/debug")
        elif isinstance(self.loggers[1].version, int):
            metrics_log_dir = Path("experiment_logs/debug")
        else:
            metrics_log_dir = (
                    Path(self.loggers[1].save_dir) / Path(self.loggers[1].name) / Path(str(self.loggers[1].version))
            )

        metrics_log_dir.mkdir(exist_ok=True)

        logger.info(f"{metrics_log_dir = }")

        fig_metrics_scalar, fig_prec_rec_crv, fig_conf, fig_conf50 = generate_test_performance_report(
            torchmetrics_results={**metrics_scalar, **metrics_tensor}, log_dir=metrics_log_dir
        )
        # quick roc curve
        roc_crv_test: torch.Tensor = self.roc_crv[2].compute()
        roc_crv_test_list = [item.tolist() for item in roc_crv_test]

        roc_curve_x = roc_crv_test_list[0][::-1]
        roc_curve_y = roc_crv_test_list[1][::-1]

        df_roc = pd.DataFrame(data={"False Positive Rate": roc_curve_x, "Sensitivity": roc_curve_y})

        fig_roc, axs, font_label, font_annots = figure_setup(num_row=1, num_cols=1, width_cm=10, height_cm=10)

        # torchmetrics doesn't return the AUC, so we need to calculate it again
        auc: float = metrics.auc(roc_curve_x, roc_curve_y)

        # don't use sort=True, this messes up the coordinates!
        sns.lineplot(
            x=roc_curve_x, y=roc_curve_y, ax=axs, sort=False, estimator=None, color="#9345c9"
        )

        axs.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

        # ghost line hack such that annotations gets automatically moved around on the plot depending on the curve
        # axs_pr.text(auc_box_str, fontsize=10,
        #            verticalalignment='top', bbox=props)
        auc_legend = plt.Line2D(
            [0],
            [0],
            color="#9345c9",
            marker="o",
            markerfacecolor="#9345c9",
            label=f"AUC: {auc:.3f}",
        )
        axs.legend(
            handles=[auc_legend],
            loc="best",
            fancybox=True,
            facecolor="#9345c9",
            shadow=False,
            framealpha=0.2,
        )

        # axs_pr.set_title("Precision-Recall Curve")
        axs.set_xlabel("False Positive Rate")
        axs.set_ylabel("Sensitivity")
        # axs_pr.grid()

        plt.tight_layout()

        self.trainer.loggers[1].experiment.log(
            {
                "test/roc_crv": wandb.Image(fig_roc),
            }
        )

        plt.show()
        # workaround for when fast_dev_run=True -> there's no wandb logger in that case, cant keep track of that otherwise
        if isinstance(self.loggers[1].version, int):
            pass
        else:
            self.trainer.loggers[1].experiment.log(
                {
                    "test/metrics": wandb.Image(fig_metrics_scalar),
                    "test/prec_rec_crv": wandb.Image(fig_prec_rec_crv),
                    "test/confusion": wandb.Image(fig_conf),
                    "test/confusion50": wandb.Image(fig_conf50),
                }
            )

        for metric in self.metrics:
            metric[2].reset()

        self.confusion[2].reset()
        self.prc_rec_crv[2].reset()
        self.confusion_50[2].reset()
        self.stat_scores[2].reset()

    def predict_step(self, batch, batch_idx) -> Any:
        inp_data, inp_data_static, labels = batch
        preds_prob, preds, labels, loss_train = self._shared_eval(batch, batch_idx, prefix="predict")
        # return self.forward(inp_data, inp_data_static)
        return preds_prob
