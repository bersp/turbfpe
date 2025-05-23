"""
TODO: Implement plot_increment_pdf.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import scipy.stats as stats

from ..utils.general import get_pdf, logspace_moving_average


def plot_stationary(data, data_split_percent):
    # split data into chunks
    n = int(np.ceil(100 / data_split_percent))
    d = data.size - n * (data.size // n)
    if d == 0:
        chunked_data = np.split(data, n)
    else:
        chunked_data = np.split(data[:-d], n)

    # calculate mean, std, skew and kurtosis for each chunk of the data
    out = np.zeros(shape=(4, n))
    for i, chunk in enumerate(chunked_data):
        out[0, i] = np.mean(chunk)
        out[1, i] = np.std(chunk)
        out[2, i] = stats.skew(chunk)
        out[3, i] = stats.kurtosis(chunk, fisher=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.set_xlabel(r"$\mathrm{percentage}~\mathrm{of}~\mathrm{data}$")

    x = np.linspace(0, 100, n, endpoint=True)
    ax1.scatter(x, out[0], label="mean")
    ax1.scatter(x, out[1], label="std")
    ax1.scatter(x, out[2], label="skewnss")
    ax1.scatter(x, out[3], label="kurtosis")

    x = np.linspace(0, 100, data.size, endpoint=True)
    ax2.plot(x, data, lw=0.1, color="k")

    ax1.legend()


def plot_pdf(data, nbins):
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlabel("$u$")
    ax.set_ylabel(r"$\mathrm{PDF}$")

    data_mean, data_std = np.mean(data), np.std(data)
    x, y = get_pdf(data, bins=nbins)

    ax.plot(
        x,
        stats.norm(data_mean, data_std).pdf(x),
        "k",
        lw=2,
        label="Gaussian distribution",
    )
    ax.plot(x, y, "o", mfc="none", mew=1.5, label="Raw data")

    ax.set_yscale("log")
    ax.grid(alpha=0.2)
    ax.legend()


def plot_spectrum(
    data, fs, int_scale, taylor_scale, comp_exponent="-5/3", ma_nbins=None
):
    # calculate and plot energy spectrum
    esd = np.abs(np.fft.rfft(data, norm="backward")) ** 2 / (data.size * fs)
    f = fs / 2 * np.linspace(0, 1, int(data.size / 2))
    # ^ same as f = np.fft.rfftfreq(data.size, d=1/fs)[1:]
    ene = 2 * esd[1:]

    ene_norm_inv = f ** (-eval(comp_exponent))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.set_xlabel(r"$f$")
    ax2.set_xlabel(r"$f$")
    ax1.set_ylabel("$E(f)$")
    ax2.set_ylabel(f"$E(f)/f^{{{comp_exponent}}}$")

    # Raw spectrum
    subsamp = np.logspace(0, np.log10(f.size), int(1e5)).astype(int)[:-1]
    ax1.plot(
        f[subsamp],
        ene[subsamp],
        c="C7",
        lw=1,
        label="Raw data",
        alpha=1,
    )
    ax2.plot(
        f[subsamp],
        ene[subsamp] * ene_norm_inv[subsamp],
        c="C7",
        lw=1,
        alpha=1,
    )

    # Smoothed (moving-average) spectrum
    if ma_nbins is not None:
        f_ma = logspace_moving_average(f, nbins=ma_nbins)
        ene_ma = logspace_moving_average(ene, nbins=ma_nbins)
        ene_norm_inv_ma = f_ma ** (-eval(comp_exponent))
        ax1.plot(f_ma, ene_ma, c="k", label="Averaged data", zorder=100)
        ax2.plot(f_ma, ene_ma * ene_norm_inv_ma, c="k", zorder=100)

        max_ene = np.max(ene_ma * ene_norm_inv_ma)
    else:
        max_ene = np.max(ene * f * ene_norm_inv)

    # Horizontal line on the spectrum max
    ax2.axhline(max_ene, ls="--", lw=2, c="#2b3339", label=f"$f^{{{comp_exponent}}}$")

    # Integral and taylor "freqs"
    ax1.axvline(1 / int_scale, lw=2, c="C1", label=r"$L^{-1}$")
    ax1.axvline(1 / taylor_scale, lw=2, c="C0", label=r"$\lambda^{-1}$")
    ax2.axvline(1 / int_scale, lw=2, c="C1")
    ax2.axvline(1 / taylor_scale, lw=2, c="C0")

    for ax in (ax1, ax2):
        ax.grid()
        ax.legend()
        ax.loglog()


def compute_taylor_scale(data, fs, ma_nbins):
    # calculate and plot energy spectrum
    esd = np.abs(np.fft.rfft(data, norm="backward")) ** 2 / (data.size * fs)
    f = fs / 2 * np.linspace(0, 1, int(data.size / 2))
    # ^ same as f = np.fft.rfftfreq(data.size, d=1/fs)[1:]
    ene = 2 * esd[1:]
    f = f[1:]
    ene = ene[1:]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel("$f$")
    ax.set_ylabel("$E$")

    # plot raw data
    subsamp = np.logspace(0, np.log10(f.size), int(1e5)).astype(int)
    ax.plot(f[subsamp], ene[subsamp], c="C7", lw=1, label="Raw data", alpha=1)

    # calculate and plot moving average energy spectrum
    if ma_nbins is not None:
        f_ma = logspace_moving_average(f, nbins=ma_nbins)
        ene_ma = logspace_moving_average(ene, nbins=ma_nbins)
        ax.plot(f_ma, ene_ma, "k", label="Averaged data")

    # mean line and text
    mean_val = np.mean(ene[(f > 1) & (f < 10)])
    mean_line = ax.axhline(
        mean_val, ls="-", color="C3", label="$E(0)$ extrapolation", lw=2.5
    )
    mean_text = ax.text(
        0.65,
        0.05,
        f"$E(0)$ = {mean_val:.2g}",
        transform=ax.transAxes,
        fontsize=14,
        va="center",
    )

    # add sliders
    fmin_line = ax.axvline(1, ls="-", c="C0")
    fmin_slider = mpl_utils.add_horizontal_slider(
        ax=ax,
        y_distance=0.03,
        label="Min freq",
        valmin=np.log(f[0]),
        valmax=np.log(f[-1]),
        valstep=f[2] - f[1],
        valinit=0.02,
        color="C0",
    )

    def fmin_update(val):
        val = np.exp(val)
        fmin_line.set_xdata([val, val])
        fmin_slider.valtext.set_text(f"{val:.2g}")

        mval = np.nanmean(
            ene[(f > np.exp(fmin_slider.val)) & (f < np.exp(fmax_slider.val))]
        )
        mean_line.set_ydata([mval, mval])
        mean_text.set_text(f"E(0) = {mval:.2g}")

    fmin_slider.on_changed(fmin_update)

    fmax_line = ax.axvline(10, ls="-", c="C1")
    fmax_slider = mpl_utils.add_horizontal_slider(
        ax=ax,
        y_distance=0.0,
        label="Max freq",
        valmin=np.log(f[0]),
        valmax=np.log(f[-1]),
        valstep=f[2] - f[1],
        valinit=np.log(10),
        color="C1",
    )

    def fmax_update(val):
        val = np.exp(val)
        fmax_line.set_xdata([val, val])
        fmax_slider.valtext.set_text(f"{val:.2g}")

        mval = np.nanmean(
            ene[(f > np.exp(fmin_slider.val)) & (f < np.exp(fmax_slider.val))]
        )
        mean_line.set_ydata([mval, mval])
        mean_text.set_text(f"E(0) = {mval:.2g}")

    fmax_slider.on_changed(fmax_update)

    ax.grid(alpha=0.2)
    ax.legend(loc="lower left")
    ax.loglog()
    ax.set_xlim(f[0], f[-1])


if __name__ == "__main__":
    partI()
