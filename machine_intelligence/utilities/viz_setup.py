import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager
import palettable
from matplotlib.gridspec import GridSpec
import numpy as np



class VariableNames:
    lookup_dict = {
        #'time_offset': 'Time Offset',
        'patientunitstayid': 'Patient Unit Stay ID',
        'noninvasivesystolic': 'Noninvasive Systolic',
        'noninvasivediastolic': 'Noninvasive Diastolic',
        'noninvasivemean': 'Noninvasive Mean',
        'cardiacoutput': 'Cardiac Output',
        'cardiacinput': 'Cardiac Input',
        'svr': 'Systemic Vascular Resistance (SVR)',
        'temperature': 'Temperature',
        'sao2': 'Oxygen Saturation (SaO2)',
        'heartrate': 'Heart Rate',
        'respiration': 'Respiration',
        'cvp': 'Central Venous Pressure (CVP)',
        'etco2': 'End-Tidal CO2 (EtCO2)',
        'systemicsystolic': 'Systemic Systolic',
        'systemicdiastolic': 'Systemic Diastolic',
        'systemicmean': 'Systemic Mean',
        'pasystolic': 'Pulmonary Artery Systolic',
        'padiastolic': 'Pulmonary Artery Diastolic',
        'pamean': 'Pulmonary Artery Mean',
        'st1': 'ST Segment 1',
        'st2': 'ST Segment 2',
        'st3': 'ST Segment 3',
        'icp': 'Intracranial Pressure (ICP)',
        'labname_-basos': 'Basophils (Basos)',
        'labname_-eos': 'Eosinophils (Eos)',
        'labname_-lymphs': 'Lymphocytes (Lymphs)',
        'labname_-monos': 'Monocytes (Monos)',
        # 'labname_-polys': 'Polymorphonuclear Leukocytes (Polys)',
        'labname_-polys': 'Polymorph. Leukocytes (Polys)',
        'labname_ALT (SGPT)': 'Alanine Aminotransferase (ALT/SGPT)',
        'labname_AST (SGOT)': 'Aspartate Aminotransferase (AST/SGOT)',
        'labname_BUN': 'Blood Urea Nitrogen (BUN)',
        'labname_Base Excess': 'Base Excess',
        'labname_CPK': 'Creatine Phosphokinase (CPK)',
        'labname_FiO2': 'Fraction of Inspired Oxygen (FiO2)',
        'labname_HCO3': 'Bicarbonate (HCO3)',
        'labname_Hct': 'Hematocrit (Hct)',
        'labname_Hgb': 'Hemoglobin (Hgb)',
        'labname_MCH': 'MC Hemoglobin (MCH)',
        'labname_MCHC': 'MC Hemoglobin Conc. (MCHC)',
        'labname_MCV': 'Mean Corpuscular Volume (MCV)',
        'labname_MPV': 'Mean Platelet Volume (MPV)',
        'labname_O2 Sat (%)': 'Oxygen Saturation (%)',
        'labname_PT': 'Prothrombin Time (PT)',
        'labname_PT - INR': 'Int. Normalized Ratio (PT-INR)',
        'labname_PTT': 'Partial Thromboplastin Time (PTT)',
        'labname_RBC': 'Red Blood Cells (RBC)',
        'labname_RDW': 'Red Cell Distribution Width (RDW)',
        'labname_Total CO2': 'Total CO2',
        'labname_WBC x 1000': 'White Blood Cell Count',
        'labname_albumin': 'Albumin',
        'labname_alkaline phos.': 'Alkaline Phosphatase',
        'labname_anion gap': 'Anion Gap',
        'labname_bedside glucose': 'Bedside Glucose',
        'labname_bicarbonate': 'Bicarbonate',
        'labname_calcium': 'Calcium',
        'labname_chloride': 'Chloride',
        'labname_creatinine': 'Creatinine',
        'labname_glucose': 'Glucose',
        'labname_lactate': 'Lactate',
        'labname_magnesium': 'Magnesium',
        'labname_pH': 'pH',
        'labname_paCO2': 'Partial Pressure of CO2 (paCO2)',
        'labname_paO2': 'Partial Pressure of O2 (paO2)',
        'labname_phosphate': 'Phosphate',
        'labname_platelets x 1000': 'Platelets Count',
        'labname_potassium': 'Potassium',
        'labname_sodium': 'Sodium',
        'labname_total bilirubin': 'Total Bilirubin',
        'labname_total protein': 'Total Protein',
        'labname_troponin - I': 'Troponin I',
        # 'sepsis_patient': 'Sepsis Patient',
        # 'sepsis_present': 'Sepsis Present',
        # 'sample_idx_6_h': 'Sample Index (6 hours)',
        # 'sepsis_sample': 'Sepsis Sample',
        # 'sepsis_sample_shifted': 'Sepsis Sample (Shifted)',
        # 'patientunitstayid': 'Patient Unit Stay ID',
        'gender': 'Sex',
        'aids': 'AIDS',
        'hepaticfailure': 'Hepatic Failure',
        'lymphoma': 'Lymphoma',
        'metastaticcancer': 'Metastatic Cancer',
        'leukemia': 'Leukemia',
        'immunosuppression': 'Immunosuppression',
        'cirrhosis': 'Cirrhosis',
        'diabetes': 'Diabetes',
        'admissionweight': 'Admission Weight',
        'admissionheight': 'Admission Height',
        'age': 'Age'
    }

    @classmethod
    def get_pretty_name(cls, variable_name):
        return cls.lookup_dict.get(variable_name, variable_name)

    @classmethod
    def get_pretty_dataframe(cls, df):
        return df.rename(columns=cls.lookup_dict)

    @classmethod
    def get_human_readable_index(cls, df):
        return df.rename(index=cls.lookup_dict)
    @classmethod
    def sort_dataframe(cls, df):
        # Sort the columns alphabetically
        df_sorted = df.sort_index(axis=1)
        # Sort the index alphabetically
        df_sorted = df_sorted.sort_index(axis=0)
        return df_sorted

    @classmethod
    def get_lab_related_names(cls):
        return {k: v for k, v in cls.lookup_dict.items() if 'labname' in k}

    @classmethod
    def get_vital_sign_names(cls):
        non_vital_signs = {
            "time_offset",
            'sepsis_patient',
            'sepsis_present',
            'sample_idx_6_h',
            'sepsis_sample',
            'sepsis_sample_shifted',
            'patientunitstayid',
            'gender',
            'aids',
            'hepaticfailure',
            'lymphoma',
            'metastaticcancer',
            'leukemia',
            'immunosuppression',
            'cirrhosis',
            'diabetes',
            'admissionweight',
            'admissionheight',
            'age'
        }
        return {k: v for k, v in cls.lookup_dict.items() if 'labname' not in k and k not in non_vital_signs}


def insert_line_breaks_in_columns(df, max_length=20):
    def insert_line_breaks(label, max_length=20):
        return '\n'.join([label[i:i+max_length] for i in range(0, len(label), max_length)])

    df.columns = [insert_line_breaks(col, max_length) for col in df.columns]
    return df



def cm2inch(value):
    """converts cm to inch for plotting
    :param float value:
    """
    return value / 2.54


def get_color_palette(number_of_colors, color_offset: int = 1):
    palettes = {
        5: palettable.matplotlib.Plasma_6.mpl_colors[color_offset:],
        6: palettable.matplotlib.Plasma_7.mpl_colors[color_offset:],
        7: palettable.matplotlib.Plasma_8.mpl_colors[color_offset:],
        8: palettable.matplotlib.Plasma_9.mpl_colors[color_offset:],
        9: palettable.matplotlib.Plasma_10.mpl_colors[color_offset:],
        10: palettable.matplotlib.Plasma_11.mpl_colors[color_offset:],
        11: palettable.matplotlib.Plasma_12.mpl_colors[color_offset:],
        12: palettable.matplotlib.Plasma_13.mpl_colors[color_offset:],
        13: palettable.matplotlib.Plasma_14.mpl_colors[color_offset:],
        14: palettable.matplotlib.Plasma_15.mpl_colors[color_offset:],
        15: palettable.matplotlib.Plasma_16.mpl_colors[color_offset:],
        16: palettable.matplotlib.Plasma_17.mpl_colors[color_offset:],
        17: palettable.matplotlib.Plasma_18.mpl_colors[color_offset:],
        18: palettable.matplotlib.Plasma_19.mpl_colors[color_offset:],
        19: palettable.matplotlib.Plasma_20.mpl_colors[color_offset:],
    }
    return palettes.get(number_of_colors,
                        palettable.matplotlib.Plasma_10.mpl_colors)  # Default to Plasma if index not found

def five_min_to_hours(x, pos):
    total_seconds = 5 * 60 * int(x)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60

    # return f"{hours}:{minutes:02d}"
    return f"{hours}"


def figure_setup(num_row=1, num_cols=1, width_cm=16, height_cm=5, share_x=False, share_y=False,
                 font_size=10, num_colors: int = 10, color_offset: int = 1, palette: palettable or None = None,  **kwargs) -> (
        plt.Figure, plt.Axes, plt.rc):
    # plot setup
    font_name = "TUM Neue Helvetica"

    sns.set(style="ticks", font=font_name)

    font_size_modifier = 0.5
    font_size_labels = 16 * 0.5
    font_size_annotations = 14 * 0.5

    font_labels = font_manager.FontProperties(family=font_name,
                                              style="normal",
                                              size=16 * font_size_modifier)
    font_annotations = font_manager.FontProperties(family=font_name,
                                                   style="normal",
                                                   size=14 * font_size_modifier)

    plt.rcParams.update({
        "font.family": font_name,
        "font.size": font_size,
        "axes.titlesize": font_size_labels,  # Title font size
        "axes.labelsize": font_size_labels,  # X and Y axis labels font size
        "xtick.labelsize": font_size_annotations,  # X tick labels font size
        "ytick.labelsize": font_size_annotations,
        "axes.grid": True
    })

    fig, axes = plt.subplots(
        nrows=num_row,
        ncols=num_cols,
        dpi=300,
        figsize=(cm2inch(width_cm), cm2inch(height_cm)),
        squeeze=True,
        sharex=share_x,
        sharey=share_y,
    )

    if isinstance(axes, plt.Axes):
        # color_palette = palettable.matplotlib.Plasma_7.mpl_colors[1:]
        # color_palette = palettable.cartocolors.qualitative.Pastel_6.mpl_colors
        # axes.set_prop_cycle("color", palettable.cartocolors.qualitative.Pastel_6.mpl_colors)

        axes.tick_params(axis="both", which="major", labelsize=font_size_labels)

        if palette is None:

            axes.set_prop_cycle("color", get_color_palette(num_colors, color_offset))
        else:

            axes.set_prop_cycle("color", palette)

        sns.despine(ax=axes)
        axes.set_axisbelow(True)

    else:
        for axs in axes.flatten():
            # axs.set_prop_cycle("color",palettable.cartocolors.qualitative.Pastel_6.mpl_colors)

            axs.tick_params(axis="both", which="major", labelsize=font_size_labels)

            if palette is None:
                axs.set_prop_cycle("color", get_color_palette(num_colors, color_offset))
            else:
                axs.set_prop_cycle("color", palette)

            sns.despine(ax=axs)
            axs.set_axisbelow(True)

    return fig, axes, font_labels, font_annotations


def figure_setup_grid(num_row=1, num_cols=1, width_ratios: list = [20, 1], height_ratios: list = [1, 1], width_cm=16,
                      height_cm=5,
                      font_size=10, num_colors: int = 10, color_offset: int = 1, **kwargs) -> (
        plt.Figure, GridSpec, list, plt.rc):
    # plot setup
    font_name = "TUM Neue Helvetica"

    sns.set(style="ticks", font=font_name)

    font_size_modifier = 0.5
    font_size_labels = 16 * 0.5
    font_size_annotations = 14 * 0.5

    font_labels = font_manager.FontProperties(family=font_name,
                                              style="normal",
                                              size=16 * font_size_modifier)
    font_annotations = font_manager.FontProperties(family=font_name,
                                                   style="normal",
                                                   size=14 * font_size_modifier)

    plt.rcParams.update({
        "font.family": font_name,
        "font.size": font_size,
        "axes.titlesize": font_size_labels,  # Title font size
        "axes.labelsize": font_size_labels,  # X and Y axis labels font size
        "xtick.labelsize": font_size_annotations,  # X tick labels font size
        "ytick.labelsize": font_size_annotations,
        "axes.grid": True
    })

    plt.clf()
    fig = plt.figure(dpi=300, figsize=(cm2inch(width_cm), cm2inch(height_cm)))
    gs = GridSpec(num_row, num_cols, figure=fig, width_ratios=width_ratios, height_ratios=height_ratios)

    axes = np.empty((num_row, num_cols), dtype=object)
    for row in range(num_row):
        for col in range(num_cols):
            ax = fig.add_subplot(gs[row, col])
            ax.tick_params(axis="both", which="major", labelsize=font_size_labels)
            ax.set_prop_cycle("color", get_color_palette(num_colors, color_offset))
            sns.despine(ax=ax)
            ax.set_axisbelow(True)
            axes[row, col] = ax

    return fig, gs, axes, font_labels, font_annotations
