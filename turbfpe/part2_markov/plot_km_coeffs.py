import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .km_coeffs_estimation_stp_opti import short_time_prop
from .markov_auxiliar_functions import get_Di_for_all_incs_and_scales


def plot_km_coeffs_estimation(km_coeffs_est_group, density_funcs_group, taylor_scale):
    u_incs = density_funcs_group.unpack("mean_per_bin0")
    D1 = km_coeffs_est_group.unpack("D1")
    D2 = km_coeffs_est_group.unpack("D2")
    D3 = km_coeffs_est_group.unpack("D3")
    D4 = km_coeffs_est_group.unpack("D4")

    scales_1d = km_coeffs_est_group.unpack("scale") / taylor_scale
    scales = np.tile(scales_1d, (u_incs.shape[1], 1)).T

    valid_idxs = km_coeffs_est_group.unpack("valid_idxs")
    u_incs[~valid_idxs] = np.nan
    D1[~valid_idxs] = np.nan
    D2[~valid_idxs] = np.nan
    D3[~valid_idxs] = np.nan
    D4[~valid_idxs] = np.nan
    scales[~valid_idxs] = np.nan

    x_label = r"$u_s / \sigma_\infty$"
    y_label = r"$s / \lambda$"

    fig, axs = plt.subplots(2, 3, subplot_kw={"projection": "3d"}, figsize=(18, 9))
    for ax in axs.flat:
        ax.invert_yaxis()
    scatter_3d(axs[0, 0], u_incs, scales, D1, x_label, y_label, r"$D^{(1)}$", alpha=0.4)
    scatter_3d(axs[0, 1], u_incs, scales, D2, x_label, y_label, r"$D^{(2)}$", alpha=0.4)
    scatter_3d(axs[1, 0], u_incs, scales, D4, x_label, y_label, r"$D^{(4)}$", alpha=0.4)
    scatter_3d(axs[1, 1], u_incs, scales, D3, x_label, y_label, r"$D^{(3)}$", alpha=0.4)

    scatter_3d(
        axs[0, 2],
        u_incs,
        scales,
        (D2**2) / D4,
        x_label,
        y_label,
        r"$(D^{(2)})^2 / D^{(4)}$",
        alpha=0.4,
    )

    scatter_3d(
        axs[1, 2],
        u_incs,
        scales,
        D4 / (D2**2),
        x_label,
        y_label,
        r"$D^{(4)} / (D^{(2)})^2$",
        alpha=0.4,
    )

    plt.show()


def plot_km_coeffs_estimation_opti(
    km_coeffs_est_group,
    density_funcs_group,
    fs,
    markov_scale_us,
    nbins,
    taylor_scale,
    taylor_hyp_vel,
):
    # No-opt
    u_incs = density_funcs_group.unpack("mean_per_bin0")
    D1 = km_coeffs_est_group.unpack("D1")
    D2 = km_coeffs_est_group.unpack("D2")

    scales_1d = km_coeffs_est_group.unpack("scale") / taylor_scale
    scales = np.tile(scales_1d, (u_incs.shape[1], 1)).T

    valid_idxs = km_coeffs_est_group.unpack("valid_idxs")
    u_incs[~valid_idxs] = np.nan
    D1[~valid_idxs] = np.nan
    D2[~valid_idxs] = np.nan
    scales[~valid_idxs] = np.nan

    x_label = r"$u_s / \sigma_\infty$"
    y_label = r"$s / \lambda$"

    # Opt
    u_incs_opti, scales_opti, D1_opti, _, D2_opti, _ = get_Di_for_all_incs_and_scales(
        density_funcs_group,
        km_coeffs_est_group,
        nbins=nbins,
        taylor_scale=taylor_scale,
        use_Di_opti=True,
    )

    fig = plt.figure(figsize=(10, 9))
    gs = gridspec.GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")
    ax3 = fig.add_subplot(gs[1, :])
    ax1.invert_yaxis()
    ax2.invert_yaxis()
    ax3.set_aspect(1)

    scatter_3d(
        ax1,
        u_incs,
        scales,
        D1,
        x_label,
        y_label,
        r"$D^{(1)}$",
        legend="No opt",
        alpha=0.4,
    )
    scatter_3d(
        ax1,
        u_incs_opti,
        scales_opti,
        D1_opti,
        x_label,
        y_label,
        r"$D^{(1)}$",
        legend="Opt",
        color="C0",
        alpha=0.4,
    )

    scatter_3d(
        ax2,
        u_incs,
        scales,
        D2,
        x_label,
        y_label,
        r"$D^{(2)}$",
        legend="No opt",
        alpha=0.4,
    )
    scatter_3d(
        ax2,
        u_incs_opti,
        scales_opti,
        D2_opti,
        x_label,
        y_label,
        r"$D^{(2)}$",
        legend="Opt",
        color="C0",
        alpha=0.4,
    )

    _plot_pdf_estimation_opti(
        ax3,
        density_funcs_group,
        km_coeffs_est_group,
        1,
        fs,
        markov_scale_us,
        taylor_scale,
        taylor_hyp_vel,
    )

    plt.show()


def _plot_pdf_estimation_opti(
    ax,
    density_funcs_group,
    km_coeffs_est_group,
    scale_idx,
    fs,
    markov_scale_us,
    taylor_scale,
    taylor_hyp_vel,
):
    km_coeffs_est = km_coeffs_est_group[scale_idx]
    dens_funcs = density_funcs_group[scale_idx]

    scale_us = km_coeffs_est.scale_us
    scale_short_us = km_coeffs_est.scale_short_us

    D1 = km_coeffs_est.D1
    D2 = km_coeffs_est.D2
    D1_opti = km_coeffs_est.D1_opti
    D2_opti = km_coeffs_est.D2_opti

    valid_idxs = km_coeffs_est.valid_idxs
    mean_per_bin0 = dens_funcs.mean_per_bin0
    mean_per_bin1 = dens_funcs.mean_per_bin1

    P_exp = dens_funcs.P_1I0

    P_exp_for_lavels = P_exp.copy()
    P_exp_for_lavels[P_exp_for_lavels < 0.15] = np.nan
    P_exp_for_lavels = np.clip(P_exp_for_lavels, None, 3 * np.nanstd(P_exp_for_lavels))
    levels = (
        np.round(
            np.logspace(
                np.log10(0.01 * np.nanmax(P_exp_for_lavels) * 1e6),
                np.log10(0.90 * np.nanmax(P_exp_for_lavels) * 1e6),
                10,
            )
        )
        / 1e6
    )

    P_opt = short_time_prop(
        y=mean_per_bin0[valid_idxs],
        x=mean_per_bin1[valid_idxs],
        scale_short_us=scale_short_us,
        scale_us=scale_us,
        D1=D1_opti,
        D2=D2_opti,
        taylor_scale=taylor_scale,
        taylor_hyp_vel=taylor_hyp_vel,
        fs=fs,
    )

    P_noopt = short_time_prop(
        y=mean_per_bin0[valid_idxs],
        x=mean_per_bin1[valid_idxs],
        scale_short_us=scale_short_us,
        scale_us=scale_us,
        D1=D1[valid_idxs],
        D2=D2[valid_idxs],
        taylor_scale=taylor_scale,
        taylor_hyp_vel=taylor_hyp_vel,
        fs=fs,
    )

    s = scale_us / markov_scale_us
    s_short = scale_short_us / markov_scale_us
    ax.set_xlabel(rf"$u_{{s={s:.2g}\Delta_\text{{EM}}}} / \sigma_\infty$")
    ax.set_ylabel(rf"$u_{{s={s_short:.2g}\Delta_\text{{EM}}}} / \sigma_\infty$")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2, 2)

    ax.contour(
        mean_per_bin0,
        mean_per_bin1,
        P_exp,
        levels=levels,
        colors="k",
        linewidths=2,
        linestyles=":",
    )
    ax.contour(
        mean_per_bin0[valid_idxs],
        mean_per_bin1[valid_idxs],
        P_noopt,
        levels=levels,
        colors="k",
        linewidths=2,
        alpha=0.8,
    )
    ax.contour(
        mean_per_bin0[valid_idxs],
        mean_per_bin1[valid_idxs],
        P_opt,
        levels=levels,
        colors="C0",
        linewidths=2,
    )

    # legend
    legend_elements = [
        Line2D([0], [0], color="k", lw=2, ls=":", label=r"$p_\text{exp}$"),
        Line2D([0], [0], color="gray", lw=2, label=r"$p_\text{stp}$"),
        Line2D([0], [0], color="C0", lw=2, label=r"$p_\text{stp}^\text{opt}$"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")


def plot_km_coeffs_fit(
    km_coeffs,
    density_funcs_group,
    km_coeffs_est_group,
    nbins,
    taylor_scale,
    use_Di_opti=True,
):
    # Rebuild the data
    u_incs, scales, D1, _, D2, _ = get_Di_for_all_incs_and_scales(
        density_funcs_group,
        km_coeffs_est_group,
        nbins,
        taylor_scale,
        use_Di_opti=use_Di_opti,
    )

    fig, (ax1, ax2) = plt.subplots(
            1, 2, subplot_kw={"projection": "3d",  "computed_zorder": False}, figsize=(14, 6)
    )
    ax1.invert_yaxis()
    ax2.invert_yaxis()

    x_label = r"$u_s / \sigma_\infty$"
    y_label = r"$s / \lambda$"


    x_lin = np.linspace(np.nanmin(u_incs), np.nanmax(u_incs), 60)
    y_lin = np.logspace(np.log10(np.nanmin(scales)), np.log10(np.nanmax(scales)), 60)
    X, Y = np.meshgrid(x_lin, y_lin)

    D1_fit = km_coeffs.eval_D1(X, Y)
    ax1.plot_surface(X, Y, D1_fit, alpha=.8, cmap="GnBu_r")
    scatter_3d(ax1, u_incs, scales, D1, x_label, y_label, r"$D^{(1)}$")


    D2_fit = km_coeffs.eval_D2(X, Y)
    ax2.plot_surface(X, Y, D2_fit, alpha=.8, cmap="GnBu_r")
    scatter_3d(ax2, u_incs, scales, D2, x_label, y_label, r"$D^{(2)}$")

    plt.show()


def scatter_3d(ax, x, y, z, x_label, y_label, z_label, legend=None, **kwargs):
    if "color" not in kwargs.keys():
        kwargs["color"] = "k"
    if legend is not None:
        kwargs["label"] = legend

    x_flat = x.flatten()
    y_flat = y.flatten()
    z_flat = z.flatten()

    mask = ~np.isnan(x_flat) & ~np.isnan(y_flat) & ~np.isnan(z_flat)
    ax.scatter(
        x_flat[mask],
        y_flat[mask],
        z_flat[mask],
        ec="white",
        s=40,
        lw=1,
        **kwargs,
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_zlabel(z_label)

    if legend:
        ax.legend()
