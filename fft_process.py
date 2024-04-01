from scipy.fft import fftn, ifftn
from scipy.fftpack import dctn, idctn
from scipy.fftpack import dct, idct
import pywt
from pywt import dwtn, idwtn
import numpy as np
import copy as cp


def process_video(video_data, X, Y, F, pert):

    s = list(video_data.shape)
    video_len = s[0]
    s[0] = max(video_len, max(F) + 1)
    fft_transform = fftn(video_data, s=s)

    f_grid, x_grid, y_grid = np.meshgrid(F, X, Y, indexing='ij')
    fft_transform[f_grid, x_grid, y_grid] += pert

    processed_data = np.abs(ifftn(fft_transform))
    processed_data = processed_data[:video_len]

    # Calculate the min and max of each frame
    min_vals = processed_data.min(axis=(1, 2), keepdims=True)
    max_vals = processed_data.max(axis=(1, 2), keepdims=True)
    # Normalize each frame
    processed_data = 255 * (processed_data - min_vals) / (max_vals - min_vals)
    # Ensure correct datatype for image data
    processed_data = processed_data.astype('uint8')

    # print(f"diff norm {np.linalg.norm(video_data-processed_data)}")

    return processed_data


def process_video_DCT(video_data, X, Y, F, pert):

    s = list(video_data.shape)
    video_len = s[0]
    s[0] = max(video_len, max(F) + 1)
    dct_transform = dctn(video_data, shape=s, norm='ortho')

    f_grid, x_grid, y_grid = np.meshgrid(F, X, Y, indexing='ij')
    dct_transform[f_grid, x_grid, y_grid] += pert

    processed_data = idctn(dct_transform, norm='ortho')
    processed_data = processed_data[:video_len]

    # # Calculate the min and max of each frame
    # min_vals = processed_data.min(axis=(1, 2), keepdims=True)
    # max_vals = processed_data.max(axis=(1, 2), keepdims=True)
    # # Normalize each frame
    # processed_data = 255 * (processed_data - min_vals) / (max_vals - min_vals)
    # # Ensure correct datatype for image data
    # processed_data = processed_data.astype('uint8')

    processed_data = np.clip(processed_data, 0, 255)
    processed_data = processed_data.astype('uint8')

    # print(np.abs(dctn(processed_data, shape=s, norm='ortho') - dct_transform).sum())
    # pdb.set_trace()

    return processed_data


def process_video_DWT(video_data, X, Y, F, pert, mode='ddd', wavelet='db1', poison_span=1/4):

    video_len, H, W = video_data.shape

    # DWT
    dwt_transform = dwtn(video_data, wavelet=wavelet, axes=(0, 1, 2))
    length = dwt_transform['aaa'].shape[0]
    start = int((1 - poison_span)/2 * length)
    end = start + int(poison_span * length)


    for key in dwt_transform:

        dwt_transform[key][start:end] += pert

    processed_data = idwtn(dwt_transform, wavelet=wavelet, axes=(0, 1, 2))
    processed_data = processed_data[:video_len]

    processed_data = np.clip(processed_data, 0, 255)
    processed_data = processed_data.astype('uint8')


    return processed_data


def process_video_DST(video_data, X, Y, F, pert):
    return



