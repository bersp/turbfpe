import matplotlib.pyplot as plt
import numpy as np

from ..utils.general import get_pdf


def plot_entropy(entropies, nbins):
    medium_entropy = entropies.medium_entropy
    system_entropy = entropies.system_entropy
    total_entropy = entropies.total_entropy

    is_nan = np.isnan(total_entropy)
    medium_entropy = medium_entropy[~is_nan]
    system_entropy = system_entropy[~is_nan]
    total_entropy = total_entropy[~is_nan]

    # Create one figure with two subplots side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    hist_min = np.nanmin([medium_entropy, system_entropy, total_entropy])
    hist_max = np.nanmax([medium_entropy, system_entropy, total_entropy])
    bins = np.linspace(hist_min, hist_max, nbins)

    total_entropy, idxs = _filter_data_using_min_events(
        total_entropy, bins, min_events=2
    )
    medium_entropy = medium_entropy[idxs]
    system_entropy = system_entropy[idxs]

    hist_min = np.nanmin([medium_entropy, system_entropy, total_entropy])
    hist_max = np.nanmax([medium_entropy, system_entropy, total_entropy])
    bins = np.linspace(hist_min, hist_max, nbins)

    centers_med, pdf_med = get_pdf(medium_entropy, bins, density=True)
    centers_sys, pdf_sys = get_pdf(system_entropy, bins, density=True)
    centers_tot, pdf_tot = get_pdf(total_entropy, bins, density=True)

    _, counts_med = get_pdf(medium_entropy, bins, density=False)
    _, counts_sys = get_pdf(system_entropy, bins, density=False)
    _, counts_tot = get_pdf(total_entropy, bins, density=False)

    # Plot pdf
    ax1.plot(centers_med, pdf_med, label=r"$\Delta S_{\mathrm{med}}$", linewidth=1.5)
    ax1.plot(centers_sys, pdf_sys, label=r"$\Delta S_{\mathrm{sys}}$", linewidth=1.5)
    ax1.plot(
        centers_tot, pdf_tot, "--k", label=r"$\Delta S_{\mathrm{tot}}$", linewidth=1.5
    )

    # Frecuency ax
    ax1_twinx = ax1.twinx()
    ax1_twinx.plot(centers_tot, counts_tot, alpha=0)
    ax1_twinx.set_yscale("log")
    ax1_twinx.set_ylabel(r"$\mathrm{Frequency}$")

    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.4)
    ax1.set_xlabel(r"$\mathrm{Entropy}$")
    ax1.set_ylabel(r"$\mathrm{PDF}$")
    mean_ds = np.nanmean(total_entropy)
    ax1.set_title(r"$\langle \Delta S_{\mathrm{tot}}\rangle = $" f"{mean_ds:0.2f}")
    ax1.axvline(0, color="k", linestyle="-")
    ax1.legend()

    # IFT
    n_samps = 20
    sample_sizes = np.round(
        np.logspace(1, np.log10(total_entropy.size), n_samps)
    ).astype(int)
    ft_values = np.empty(n_samps)
    ft_errors = np.empty(n_samps)
    for i, size in enumerate(sample_sizes):
        slice_data = np.exp(-total_entropy[:size])
        ft_values[i] = np.nanmean(slice_data)
        ft_errors[i] = np.nanstd(slice_data) / np.sqrt(size)

    ax2.errorbar(
        sample_sizes,
        ft_values,
        yerr=ft_errors,
        fmt="ok",
        linewidth=1.2,
        label=r"$\langle e^{-\Delta S_{\mathrm{tot}}}\rangle_N$",
    )
    ax2.axhline(1.0, color="k", linestyle="--", linewidth=1.0, label="ift = 1")
    ax2.set_xscale("log")
    ax2.grid(True, alpha=0.4)

    ax2.set_xlabel(r"$N$")
    ax2.set_ylabel(r"$\langle e^{-\Delta S_{\mathrm{tot}}}\rangle_N$")
    mean_val = np.nanmean(np.exp(-total_entropy))
    ax2.set_title(
        r"$\langle e^{-\Delta S_{\mathrm{tot}}}\rangle_{\max(N)} =$ "
        f"{mean_val:0.2f}",
        fontsize=12,
    )
    ax2.legend()

    # DFT
    bound = np.max(np.abs(total_entropy))
    bins = np.linspace(-bound, bound, 151, endpoint=True)
    centers_tot, pdf_tot = get_pdf(total_entropy, bins=bins)
    ent_abs_l, dft_l = [], []

    centers_positive = centers_tot[centers_tot.size // 2 :]
    pdf_positive = pdf_tot[centers_tot.size // 2 :]

    # centers_negative = centers_tot[: centers_tot.size // 2]
    pdf_negative = pdf_tot[: centers_tot.size // 2]

    pdf_positive[pdf_positive == 0] = np.nan
    pdf_negative[pdf_negative == 0] = np.nan

    for i in range(len(centers_positive)):
        ent_abs_l.append(centers_positive[i])
        dft = np.log(pdf_positive[i] / pdf_negative[-i - 1])
        dft_l.append(dft)

    ax3.plot(ent_abs_l, ent_abs_l, "--k", label="DFT")
    ax3.plot(ent_abs_l, dft_l, "o", mfc="none")

    ax3.set_xlabel(r"$|\Delta S_{\mathrm{tot}}|$")
    ax3.set_ylabel(r"$\ln(p(\Delta S_{\mathrm{tot}}) / p(-\Delta S_{\mathrm{tot}}))$")
    ax3.legend()


def _filter_data_using_min_events(data, bins, min_events):
    x, counts = get_pdf(data, bins, density=False)

    mask = counts < min_events
    lb_idx = np.where(mask[: mask.size // 2])[0][-1]
    ub_idx = np.where(mask[mask.size // 2 :])[0][0] + mask.size // 2

    idxs = (x[lb_idx] <= data) & (data <= x[ub_idx])

    return data[idxs], idxs
