import matplotlib.pyplot as plt
import numpy as np

from ..utils.general import logspace_moving_average


def calc_int_length(data, fs):
    if data.size > int(1e5):
        data = data[: int(2 + 1e5)]

    # calculate and plot energy spectrum
    esd = np.abs(np.fft.rfft(data, norm="backward")) ** 2 / (data.size * fs)
    f = fs / 2 * np.linspace(0, 1, int(data.size / 2))
    # ^ same as f = np.fft.rfftfreq(data.size, d=1/fs)[1:]
    ene = 2 * esd[1:]
    f = f[1:]
    ene = ene[1:]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlabel("f")
    ax.set_ylabel("E")

    # plot raw data
    subsamp = np.logspace(0, np.log10(f.size), int(1e5)).astype(int)
    ax.plot(f[subsamp], ene[subsamp], c="C7", lw=1, label="Raw data", alpha=1)

    ma_bins = 1000
    f_ma = logspace_moving_average(f, nbins=ma_bins)
    ene_ma = logspace_moving_average(ene, nbins=ma_bins)
    ax.plot(f_ma, ene_ma, "k", label="Averaged data")

    # mean line and text
    e0_val = np.mean(ene[(f > 1) & (f < 10)])
    mean_line = ax.axhline(
        e0_val, ls="-", color="C3", label="E(0) extrapolation", lw=2.5
    )
    e0_text = ax.text(
        0.65,
        0.05,
        f"E(0) = {e0_val:.3g}",
        transform=ax.transAxes,
        fontsize=14,
        va="center",
    )

    # add sliders
    fmin_line = ax.axvline(1, ls="-", c="C0")
    fmin_slider = add_horizontal_slider(
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
        global e0_val
        val = np.exp(val)
        fmin_line.set_xdata([val, val])
        fmin_slider.valtext.set_text(f"{val:.2g}")

        e0_val = np.nanmean(
            ene[(f > np.exp(fmin_slider.val)) & (f < np.exp(fmax_slider.val))]
        )
        mean_line.set_ydata([e0_val, e0_val])
        e0_text.set_text(f"E(0) = {e0_val:.3g}")

    fmin_slider.on_changed(fmin_update)

    fmax_line = ax.axvline(10, ls="-", c="C1")
    fmax_slider = add_horizontal_slider(
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
        fmax_slider.valtext.set_text(f"{val:.3g}")

        e0_val = np.nanmean(
            ene[(f > np.exp(fmin_slider.val)) & (f < np.exp(fmax_slider.val))]
        )
        mean_line.set_ydata([e0_val, e0_val])
        e0_text.set_text(f"E(0) = {e0_val:.3g}")

    fmax_slider.on_changed(fmax_update)

    ax.grid(alpha=0.2)
    ax.legend(loc="lower left")
    ax.loglog()
    ax.set_xlim(f[0], f[-1])

    plt.show()

    data_mean, data_var = np.mean(data), np.var(data)
    e0_val = np.nanmean(
        ene[(f > np.exp(fmin_slider.val)) & (f < np.exp(fmax_slider.val))]
    )
    return e0_val * data_mean / (4 * data_var)
