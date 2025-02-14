import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def mpl_setup(params=None):
    if params:
        usetex_flag = params.read("config.misc.mpl.usetex")
        constrained_layout_flag = params.read("config.misc.mpl.constrained_layout")
    else:
        usetex_flag = False
        constrained_layout_flag = False

    mpl.rcParams.update(
        {
            "text.usetex": usetex_flag,
            "text.latex.preamble": r"\usepackage{amsmath}",
            "figure.constrained_layout.use": constrained_layout_flag,
        }
    )

    mpl.rc("legend", fancybox=False)

    colors = [
        (0.2823529411764706, 0.47058823529411764, 0.8156862745098039),
        (0.9333333333333333, 0.5215686274509804, 0.2901960784313726),
        (0.41568627450980394, 0.8, 0.39215686274509803),
        (0.8392156862745098, 0.37254901960784315, 0.37254901960784315),
        (0.5843137254901961, 0.4235294117647059, 0.7058823529411765),
        (0.5490196078431373, 0.3803921568627451, 0.23529411764705882),
        (0.8627450980392157, 0.49411764705882355, 0.7529411764705882),
        (0.4745098039215686, 0.4745098039215686, 0.4745098039215686),
        (0.8352941176470589, 0.7333333333333333, 0.403921568627451),
        (0.5098039215686274, 0.7764705882352941, 0.8862745098039215),
    ]
    mpl.rc("axes", prop_cycle=mpl.cycler(color=colors))

    for tick in ["xtick", "ytick"]:
        mpl.rc(tick, labelsize=15)
        mpl.rc(f"{tick}.major", width=1, size=6)

    mpl.rc("axes", labelsize=20, linewidth=1.5)
    mpl.rc("legend", fontsize=14, fancybox=False)


def save_fig(filename):
    plt.savefig(filename, dpi=150, bbox_inches="tight", transparent=True)


def add_horizontal_slider(ax, y_distance=0.0, **slider_kwargs):
    def xaligned_axes(ax, y_distance, width):
        return plt.axes(
            [
                ax.get_position().x0,
                ax.get_position().y1 + y_distance,
                ax.get_position().width,
                width,
            ]
        )

    slider_ax = xaligned_axes(ax=ax, y_distance=y_distance, width=0.03)
    slider = Slider(ax=slider_ax, **slider_kwargs)
    slider.vline._linewidth = 0
    return slider


def add_vertical_slider(ax, x_distance=0.0, **slider_kwargs):
    def yaligned_axes(ax, x_distance, width):
        return plt.axes(
            [
                ax.get_position().x1 + x_distance,
                ax.get_position().y0,
                width,
                ax.get_position().height,
            ]
        )

    slider_ax = yaligned_axes(ax=ax, x_distance=x_distance, width=0.03)
    slider = Slider(ax=slider_ax, orientation="vertical", **slider_kwargs)
    slider.vline._linewidth = 0
    return slider
