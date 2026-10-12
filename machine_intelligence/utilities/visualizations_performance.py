import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import palettable
import matplotlib.ticker as ticker
from sklearn import metrics

from machine_intelligence.utilities.viz_setup import figure_setup


def generate_test_performance_report(torchmetrics_results: dict, log_dir: Path):
    """
    Take the results from above function and plot the values in a single report
    """

    # Todo: Overview across metrics
    logger.debug("Starting Test Report Generation!")

    results = {}
    # Todo: Conversion to numpy may be better in test_performance()
    for metric in torchmetrics_results:
        if type(torchmetrics_results[metric]) is tuple:
            # we can't convert a tuple of tensors right away
            metric_array = [
                tuple_element.detach().cpu().numpy().tolist()
                for tuple_element in torchmetrics_results[metric]
            ]
            results[metric] = metric_array

        else:
            if isinstance(torchmetrics_results[metric], float):
                results[metric] = torchmetrics_results[metric]
            else:
                results[metric] = torchmetrics_results[metric].detach().cpu().numpy().tolist()

    scalar_labels = {
        "test/acc": "Accuracy",
        "test/spec": "Specificity",
        "test/prec": "Precision",
        "test/recall@80": "Precision@Recall80",
        "test/recall": "Recall",
        "test/f1": "F1",
        "test/fpr": "False Positive Rate",
        "test/auroc": "AUROC"
    }

    scalar_results: dict = {key: results[key] for key in scalar_labels}

    scaler_results_df: pd.DataFrame = pd.DataFrame(data=scalar_results, index=[0])
    scaler_results_df.columns = scalar_labels.values()

    # Test Report Figure Setup
    fig_1, axs_1, font_label, font_annots = figure_setup(num_row=1, num_cols=1, width_cm=14, height_cm=14)
    fig_2, axs_2, _, _ = figure_setup(num_row=1, num_cols=1, width_cm=10, height_cm=10)
    fig_3, axs_3, _, _ = figure_setup(num_row=1, num_cols=1, width_cm=7, height_cm=7)
    fig_4, axs_4, _, _ = figure_setup(num_row=1, num_cols=1, width_cm=7, height_cm=7)
    color_palette = palettable.matplotlib.Plasma_9.mpl_colors[1:]

    # fig_1.suptitle("Performance Metrics on Testset")

    axs_bars = axs_1
    # bar plots
    sns.barplot(scaler_results_df, ax=axs_bars, palette=color_palette)

    for container in axs_bars.containers:
        axs_bars.bar_label(container, fontsize=10, fmt="{:.4f}")

    axs_bars.set_ylabel("Fraction")
    axs_bars.set_xticklabels(axs_bars.get_xticklabels(), rotation=20, ha="right")

    # plt.tight_layout(pad=1.2)

    # precision and recall curve
    axs_pr = axs_2

    pr_curve_x = results["test/prc_rec_crv"][1][::-1]  # recall
    pr_curve_y = results["test/prc_rec_crv"][0][::-1]  # precision

    # torchmetrics doesn't return the AUC, so we need to calculate it again
    # precision, recall, thresholds = pr_curve(pred, target)
    auc: float = metrics.auc(pr_curve_x, pr_curve_y)

    # don't use sort=True, this messes up the coordinates!
    sns.lineplot(
        x=pr_curve_x, y=pr_curve_y, ax=axs_pr, sort=False, estimator=None, color="#9345c9"
    )

    axs_pr.yaxis.set_major_locator(ticker.MultipleLocator(0.1))

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
    axs_pr.legend(
        handles=[auc_legend],
        loc="best",
        fancybox=True,
        facecolor="#9345c9",
        shadow=False,
        framealpha=0.2,
    )

    # axs_pr.set_title("Precision-Recall Curve")
    axs_pr.set_ylabel("Precision")
    axs_pr.set_xlabel("Recall")
    # axs_pr.grid()
    axs_pr.set(xlim=(0, 1), ylim=(0, 1))

    # plt.tight_layout()

    # confusion matrix
    axs_conf = axs_3
    sns.heatmap(
        results["test/confusion"],
        ax=axs_conf,
        xticklabels=["Control", "Sepsis-3"],
        yticklabels=["Control", "Sepsis-3"],
        annot=True,
        fmt="",
        annot_kws={"fontsize": 8},
        cbar=False,
    )
    axs_conf.set_xlabel("Predicted Class")
    axs_conf.set_ylabel("True Class")
    # axs_conf.set_title("Confusion Matrix")
    axs_conf.set_yticklabels(axs_conf.get_yticklabels(), rotation=90)

    # plt.tight_layout()

    # confusion matrix with 50 % threshold
    axs_conf50 = axs_4
    sns.heatmap(
        results["test/confusion50"],
        ax=axs_conf50,
        xticklabels=["Control", "Sepsis-3"],
        yticklabels=["Control", "Sepsis-3"],
        annot=True,
        fmt="",
        annot_kws={"fontsize": 8},
        cbar=False,
    )
    axs_conf50.set_xlabel("Predicted Class")
    axs_conf50.set_ylabel("True Class")
    # axs_conf.set_title("Confusion Matrix")
    axs_conf50.set_yticklabels(axs_conf.get_yticklabels(), rotation=90)

    plt.tight_layout()
    # plt.show()

    # saving of result data to csv, just to be safe or retrieve it later on

    # Todo: Saving csvs to wandb
    # scalars
    scalar_results_path = log_dir / Path("test_scalar_results.csv")
    logger.info(f"Saving scalar test results to {scalar_results_path}")
    scaler_results_df.to_csv(path_or_buf=scalar_results_path)

    # precision & recall curve
    prec_rec_crv_results_df: pd.DataFrame = pd.DataFrame(data={"x": pr_curve_x, "y": pr_curve_y},
                                                         )
    prec_rec_crv_results_path = log_dir / Path("precision_recall_curve_results.csv")
    logger.info(f"Saving scalar test results to {prec_rec_crv_results_path}")
    prec_rec_crv_results_df.to_csv(path_or_buf=prec_rec_crv_results_path)

    # confusion matrix
    conf_results_df: pd.DataFrame = pd.DataFrame(data=results["test/confusion"])
    conf_results_path = log_dir / Path("confusion_matrix_results.csv")
    logger.info(f"Saving scalar test results to {conf_results_path}")
    conf_results_df.to_csv(path_or_buf=conf_results_path)

    fig_1.savefig(log_dir / Path("performance_metrics.svg"),
                  format="svg")

    fig_2.savefig(log_dir / Path("prc_curve.svg"),
                  format="svg")

    fig_3.savefig(log_dir / Path("conf80.svg"),
                  format="svg")

    fig_4.savefig(log_dir / Path("conf50.svg"),
                  format="svg")

    return fig_1, fig_2, fig_3, fig_4
