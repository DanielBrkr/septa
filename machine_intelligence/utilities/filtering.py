import pywt
import numpy as np
import pandas as pd

def pad_signal_with_same(data, pad_length):
    """Padding to somewhat mitigate edge effects"""
    start_pad = np.full(pad_length, data[0])
    end_pad = np.full(pad_length, data[-1])
    padded_data = np.concatenate([start_pad, data, end_pad])
    return padded_data

def wavelet_denoising(data, wavelet="db5", level=1, pad_length=10):
    padded_data = pad_signal_with_same(data, pad_length)
    # coeffs = pywt.wavedec(data, wavelet, mode="per")
    coeffs = pywt.wavedec(padded_data, wavelet, mode="per")
    # sigma = (1/0.6745) * np.median(np.abs(coeffs[-level] - np.median(coeffs[-level])))
    sigma = (1 / 0.6745) * np.mean(np.abs(coeffs[-level] - np.mean(coeffs[-level])))

    if np.all(coeffs[-level] == 0):

        return data

    else:
        sigma = (1 / 0.6745) * np.mean(np.abs(coeffs[-level] - np.mean(coeffs[-level])))
        uthresh = sigma * np.sqrt(2 * np.log(len(padded_data)))

    # uthresh = sigma * np.sqrt(2 * np.log(len(data)))
    coeffs[1:] = (pywt.threshold(i, value=uthresh, mode="soft") for i in coeffs[1:])
    denoised_padded_data = pywt.waverec(coeffs, wavelet, mode="per")
    denoised_data = denoised_padded_data[pad_length:-pad_length]
    denoised_data[denoised_data < 0] = 0 # related to step changes on zero filled values (< 0 is unphysiological)
    return denoised_data

def apply_wavelet_denoising_to_dataframe(df, wavelet="db5", level=1, pad_length=10):
    denoised_df = pd.DataFrame(index=df.index)
    for column in df.columns:
        denoised_data = wavelet_denoising(df[column].to_numpy(), wavelet, level, pad_length=pad_length)
        denoised_data = denoised_data[:len(df)]
        denoised_df[column] = denoised_data
    return denoised_df
