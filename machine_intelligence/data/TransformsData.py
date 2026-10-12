import numpy as np
import pandas as pd
from scipy import signal

from machine_intelligence.utilities.filtering import apply_wavelet_denoising_to_dataframe


class ZeroFill:
    def __call__(self, sample: pd.DataFrame) -> pd.DataFrame:
        sample = sample.replace(to_replace=np.nan, value=float(0))
        return sample


class SampleAndHold:
    def __call__(self, sample):
        sample = sample.ffill()
        sample = sample.replace(to_replace=np.nan, value=float(0))

        return sample

class LowPassButter:
    # setup of the filter

    def __init__(self):
        # sampling frequency of 5 mins in Hz  -> 1 / 60 Hz
        fs = 1 / (60 * 5)

        nyquist = 0.5 * fs  # f_nyquist = 0.5 * f_sampling
        #cutoff = (1 / 60) * 59  # 1 / 60 Hz * x (empirically derived)
        cutoff = 0.001  # (1 / 60) * 55  # 1 / 60 Hz * x
        normal_cutoff = cutoff / nyquist
        # self.sos = signal.butter(1, cutoff, btype="lowpass", output="sos")
        self.sos = signal.butter(1, normal_cutoff, btype="lowpass", output="sos")


    def __call__(self, sample: pd.DataFrame) -> pd.DataFrame:

        sample_columns = sample.columns
        np_df_filtered: np.ndarray = signal.sosfiltfilt(self.sos, sample, axis=0)
        df_sample_filtered = pd.DataFrame(np_df_filtered, columns=sample_columns)

        return df_sample_filtered

class WaveletDenoising:
    """Note this works only when"""

    def __call__(self, sample: pd.DataFrame) -> pd.DataFrame:
        df_sample_filtered = apply_wavelet_denoising_to_dataframe(df=sample, wavelet="db8", level=1, pad_length=10)

        return df_sample_filtered




