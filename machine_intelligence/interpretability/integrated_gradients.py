import torch
from tint.attr import TemporalIntegratedGradients
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from captum.attr import IntegratedGradients
from loguru import logger
import seaborn as sns
import os
from pathlib import Path
import lightning as lit
from datetime import datetime

from hydra import initialize, initialize_config_module, initialize_config_dir, compose

from machine_intelligence.data.SepsisPredDataModule import SepsisPredDataModule

from machine_intelligence.models.GatedTransformer import GatedTransformer


from machine_intelligence.utilities.visualizations_interpretability import (
    plot_data_vs_attributions,
    plot_attributions_barplot,
)

from machine_intelligence.models.GatedTransformer import GatedTransformer

class ModelShapWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        # self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_ts_in, x_static_in):
        # self.model.eval()

        if isinstance(self.model, GatedTransformer):
            x, _, _, _, _, _, _, _, _ = self.model(x_ts_in, x_static_in, "test")
        else:
            x = self.model(x_ts_in, x_static_in)

        x = torch.nn.functional.sigmoid(x)

        return x

def determine_temporal_and_static_attribution(
    model,
    batch_to_explain,
    num_samples: int,
    considered_features: [str],
    considered_features_static: [str],
):
    def forward_func(inputs):
        time_series_data, static_data = inputs
        # return model(time_series_data, static_data)
        return model(time_series_data, static_data)

    model = ModelShapWrapper(model)

    # explain_train_ts, explain_train_static, _ = batch_explainer_train
    explain_test_ts, explain_test_static, _ = batch_to_explain

    if use_static_vars := True:
        # x_train_explain = [explain_train_ts, explain_train_static]

        explain_test_static_rep = torch.tile(explain_test_static, (72, 1, 1))
        explain_test_static_rep_swap = torch.swapaxes(explain_test_static_rep, 0, 1)
        explain_test = (explain_test_ts[:50], explain_test_static_rep_swap[:50])

    else:
        # x_train_explain = explain_train_ts
        x_test_explain = explain_test_ts[:50]

    # still relevant? related to an issue with cuda and captum
    # if trainer.accelerator == "cuda":
    #     torch.backends.cudnn.enabled = False
    def forward_swap_attr_static(inputs_static, inputs_ts):
        return model(inputs_ts, inputs_static)

    explainer_ts = TemporalIntegratedGradients(model)
    explainer_static = IntegratedGradients(forward_swap_attr_static)

    samples = num_samples
    _, preds_attr = torch.max(
        model(explain_test_ts[:samples], explain_test_static[:samples]), dim=1
    )

    # attributions batch size x time steps x features
    attributions_ts = explainer_ts.attribute(
        explain_test_ts[:samples],
        # target=preds_att,
        target=preds_attr,
        additional_forward_args=(explain_test_static[:samples],),
        temporal_target=False,
        show_progress=True,
    )

    attributions_static = explainer_static.attribute(
        inputs=explain_test_static[:samples],
        target=preds_attr,
        additional_forward_args=(explain_test_ts[:samples],),
    )

    # -------- plotting related

    # sample attributions

    attr_sample_ts = attributions_ts[0, :, :]
    attr_sample_static = attributions_static[0, :]

    # Todo: Plots for static attributions (determine column names etc.)

    df_sample_ts = pd.DataFrame(data=explain_test_ts[0], columns=considered_features)

    # dfs sample time series attributions
    df_sample_attr_ts = pd.DataFrame(attr_sample_ts, columns=considered_features)
    df_sample_attr_sum_ts = df_sample_attr_ts.sum(axis=0)
    df_sample_attr_cum = pd.DataFrame(attr_sample_ts.cumsum(0), columns=considered_features)

    # dfs sample static attributions
    # Todo: Add column names
    df_sample_attr_static = pd.DataFrame(
        data=attr_sample_static.unsqueeze(dim=1).T,
        columns=considered_features_static,
        index=["attribution"],
    )
    df_sample_attr_sum_static = df_sample_attr_static.sum(axis=0)

    # average time series attribution (maybe also median?)
    mean_attributions_ts = torch.mean(attributions_ts, dim=0)
    mean_attributions_static = torch.mean(attributions_static, dim=0)

    df_mean_attr_ts = pd.DataFrame(
        mean_attributions_ts,
        columns=considered_features,
    )

    df_mean_attr_static = pd.DataFrame(
        torch.unsqueeze(mean_attributions_static, dim=1).T,
        columns=considered_features_static,
        index=["attribution"],
    )
    df_mean_attr_static.sort_values(by="attribution", axis=1, ascending=False)

    # total sum attributions, should be more helpful than mean
    # Todo: Sum Attr, mean is not as helpful
    df_attr_ts_all_sum = pd.DataFrame(data=torch.sum(attributions_ts, dim=1), columns=considered_features)

    sns.heatmap(df_attr_ts_all_sum.T).set_title("Per Sample Cumulated Time Series Attributions")
    plt.tight_layout()
    plt.show()


    # --- per sample static attributions heatmap (no need to cumulate this)
    df_attr_static_all = pd.DataFrame(data=attributions_static, columns=considered_features_static)
    sns.heatmap(df_attr_static_all.T).set_title("Per Sample Cumulated Static Attributions")
    plt.tight_layout()
    plt.show()


    logger.info(f"{df_mean_attr_ts.head() = }")
    logger.info(f"{df_mean_attr_static.head() = }")

    # doesn't make too much sense actually
    # fig_mean_attr_ts = plot_attributions_barplot(
    #     df_mean_attr_ts, title="Mean Attributions Time Series"
    # )
    #
    fig_mean_attr_static = plot_attributions_barplot(
        df_mean_attr_static, title="Mean Attributions Static"
    )

    fig_attr_tig = plot_data_vs_attributions(
        df_data_ts=df_sample_ts,
        df_sample_attribution_ts=df_sample_attr_ts,
        title="Time Series Attributions (Sample)",
    )

    # fig_attr_tig_mean = plot_data_vs_attributions(
    #     df_data_ts=df_sample_ts,
    #     df_sample_attribution_ts=df_mean_attr_ts,
    #     title="Time Series Attributions (Mean)",
    # )

    fig_attr_tig_sample_sum = plot_attributions_barplot(
        df_sample_attr_sum_ts, title="Summed Attributions Time Series (Sample)"
    )

    plt.show()

    wandb.log({"interp/TIG_sample": wandb.Image(fig_attr_tig)})
    wandb.log({"interp/TIG_sample_sum": wandb.Image(fig_attr_tig_sample_sum)})

    # wandb.log({"interp/TIG_ts_mean": wandb.Image(fig_mean_attr_ts)})
    wandb.log({"interp/IG_static": wandb.Image(fig_mean_attr_static)})

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    cwd = Path.cwd()
    # project_root = Path(cwd).parents[1]
    project_root = Path(cwd)

    model_path = project_root / Path("machine_intelligence/models/trained_models/gtn_final/checkpoints/last.ckpt")
    model = GatedTransformer.load_from_checkpoint(model_path, map_location="cpu")
    # model = GatedTransformer.load_from_checkpoint(model_path)

    with initialize(version_base=None, config_path="../../configs"):
        cfg = compose(config_name="run_training.yaml", overrides=["data=cohort_a_r0.4_prev_0.3_imp_none"])

        print(cfg)


    wandb.init(mode="disabled")

    os.environ['PYTHONHASHSEED'] = str(42)
    lit.seed_everything(42, workers=True)
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    sepsis_pred_data = SepsisPredDataModule(cfg=cfg)

    sepsis_pred_data.setup(stage="fit")
    sepsis_pred_data.setup(stage="test")

    background_dl = sepsis_pred_data.train_dataloader()
    inference_dl = sepsis_pred_data.test_dataloader()

    batch_background = next(iter(background_dl))

    batch_features_ts, batch_features_stat, batch_labels = batch_background
    determine_temporal_and_static_attribution(model=model, batch_to_explain=batch_background, num_samples=1,
                                              considered_features=sepsis_pred_data.considered_features,
                                              considered_features_static=sepsis_pred_data.considered_features_static)
