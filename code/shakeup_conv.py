import numpy as np 
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter


def perform_shakeup(data_xas, data_xps):

    energy_XAS = data_xas[:, 0]
    mu = data_xas[:, 1]
    dE = energy_XAS[1] - energy_XAS[0]

    # -------------------------
    # Find edge position
    # -------------------------
    smooth_width_eV = 1.0 
    window_pts = int(round(smooth_width_eV / dE))
    if window_pts % 2 == 0:
        window_pts += 1

    mu_smooth = savgol_filter(mu, window_pts, polyorder=3)
    dmu_dE = np.gradient(mu_smooth, energy_XAS)
    E_edge = energy_XAS[np.argmax(dmu_dE)]

    # -------------------------
    # Determine asymmetric padding
    # Goal: E_edge near center of padded range
    # -------------------------
    n = len(energy_XAS)
    i_edge = np.argmin(np.abs(energy_XAS - E_edge))
    center_index = n // 2

    # Number of points to pad on each side
    pad_left = max(0, 2*(center_index - i_edge))
    pad_right = max(0, 2*(i_edge - center_index))
    
    # -------------------------
    # Build padded energy grid
    # -------------------------
    energy_pad = (
        energy_XAS[0] - pad_left * dE
        + np.arange(n + pad_left + pad_right) * dE
    )

    mu_pad = np.pad(mu, (pad_left, pad_right),
                     mode="constant", constant_values=0.0)

    # -------------------------
    # Build shake-up kernel on ω grid
    # -------------------------
    omega_pad = energy_pad - E_edge

    inter_xps = interp1d(
        -data_xps[:, 0],      # ω grid of XPS (QP peak at ω = 0)
        data_xps[:, 1],
        kind="linear",
        bounds_error=False,
        fill_value=0.0
    )

    A_pad = inter_xps(omega_pad)

    # -------------------------
    # Convolution
    # -------------------------
    conv_pad = np.convolve(mu_pad, A_pad, mode="same") * dE

    # -------------------------
    # Remove padding, return on original grid
    # -------------------------
    conv = conv_pad[pad_left:pad_left + n]

    return conv


def calc_scale(data_exp, data_simu, shift, xmin, xmax): 
    exp_indices = np.logical_and(data_exp[:,0] > xmin, data_exp[:,0] < xmax)
    simu_indices = np.logical_and(data_simu[:,0] > xmin-shift, data_simu[:,0] < xmax-shift)
    exp_area = np.sum((data_exp[exp_indices][1:,0] - data_exp[exp_indices][:-1,0]) * \
                      (data_exp[exp_indices][1:,1] + data_exp[exp_indices][:-1,1]) / 2)
    simu_area = np.sum((data_simu[simu_indices][1:,0] - data_simu[simu_indices][:-1,0]) * \
                        (data_simu[simu_indices][1:,1] + data_simu[simu_indices][:-1,1]) / 2)    
    return exp_area/simu_area

    