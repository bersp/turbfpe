# — `turbfpe`
This project is a pure-Python re-implementation of [**OPEN_FPE_IFT**](https://github.com/andre-fuchs-uni-oldenburg/OPEN_FPE_IFT), the open-source toolkit created by the research group Turbulence, Wind energy and Stochastics (TWiSt) at the Carl von Ossietzky University of Oldenburg (https://uol.de/en/physics/twist).
Like the original MATLAB package, it provides an end-to-end pipeline for turbulent time-series: classical statistics, Markov-property tests, Kramers–Moyal/Fokker–Planck coefficient estimation, and entropy/fluctuation-theorem analysis — now written entirely with standard scientific Python libraries.

For the scientific background and reference algorithms see:
A. Fuchs *et al.*, “An open source package to perform basic and advanced statistical analysis of turbulence data and other complex systems” *Phys. Fluids* 34, 101801 (2022). https://doi.org/10.1063/5.0107974.


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
    - [Routine flow](#routine-flow)
    - [Configuration parameters](#general-parameters)
    - [General parameters](#general-parameters)
    - [Routine parameters](#routine-parameters)
       * [Part 1 — Turbulence-analysis](#part1)
       * [Part 2 — Markov](#part2)
       * [Part 3 — Entropy production](#part3)

# Installation
```sh
git clone https://github.com/bersp/turbfpe
cd turbfpe
python -m pip install -e .
```
The `-e` flag installs the package in editable mode, meaning Python imports the code directly from this folder. Any edits you make are therefore picked up instantly, which is convenient if you want to browse the source, modify functions, or extend the library without reinstalling after every change.

# Usage
For a concrete walkthrough, see the **`analysis_example/`** directory.  
It holds the filtered velocity data set from the reference study (`Renner_8000Hz_filter.npy`), a complete parameter file (`Renner_params.toml`), and a short driver script (`analysis.py`).  The script parses the TOML file, executes the functions listed under `rutine`, and writes results to the locations defined in the configuration tables.

For a description of the parameters and each routine entry, consult the corresponding sections in the documentation.

# Documentation

## Routine flow and `params.toml`
All analysis settings live in a single **TOML** file—commonly called `params.toml`, though any filename can be used.
Parameters in this file fall into three broad categories: **configuration parameters** (paths and plotting options), **general parameters** (measurement-wide constants such as the sampling frequency, integral scale, or Taylor microscale),
and **function-specific parameters** (settings that belong to one routine only, defined in the corresponding section (e.g. p2.compute_wilcoxon_test.\*)).

By design, **the rutine section organizes the workflow** into four ordered blocks: **preanalysis**, **turbulence diagnostics**, **Markov analysis**, and **entropy production**. Each list entry is a literal function name; at runtime the driver reads these strings and calls the corresponding Python function in sequence. Leaving all entries intact executes the full end-to-end pipeline, but you can quickly comment-out or remove individual lines (or whole sub-lists) to skip stages, rerun only a subset, or pause after a checkpoint and experiment with the outputs before continuing.

## Configuration parameters

#### Data input
- `data.io.path` — str. Path to the *.npy* file containing the raw signal.  
  The array may be 1-D (a single series) or 2-D, where each row is an independent realisation and each column is a sample along the primary axis (time, space, etc.).
- `data.prop_to_use` — either a float [0 – 1] (for 1-D data) or a two-floats list `[[0 – 1], [0 – 1]]` (for 2-D data). Specifies the fraction(s) of the data wanted to load. Useful to do small tests.

- `data.opts.norm_data` — bool. Divide the data by $\sqrt{2}\,\sigma$ (see `data.stats.std`).
- `data.opts.flip_data` — bool. Reverse the time axis when `true`.
- `data.shape` — Tracks the shape of the input data (read-only).

#### Configuration keys
- `config.io.processed_data_save_path` — str. Directory for intermediate *.npy / *.npz* files.
- `config.io.figures_save_path` — str. Directory where generated figures are written.
- `config.io.save_filenames_prefix` — str. Prefix prepended to every saved file.

- `config.mpl.usetex` — bool. Activate LaTeX text-rendering in Matplotlib.
- `config.mpl.constrained_layout` — bool. Use Matplotlib’s *constrained-layout* engine.
- `config.mpl.show_figures` — bool. Display figures interactively.
- `config.mpl.save_figures` — bool. Write figures to `figures_save_path`.

#### Stats (read-only, filled during pre-analysis)
- `stats.mean` — float. Record mean.
- `stats.rms` — float. Root-mean-square value.
- `stats.std` — float. Standard deviation $\sigma$.
- `stats.skew` — float. Skewness.
- `stats.kurtosis` — float. Kurtosis (non-Fisher).
- `stats.range` — float. Peak-to-peak amplitude.

> The `data.stats.*` entries are computed during pre-analysis by **`p0.compute_and_write_data_stats`**.


## General parameters
- `data` — Input signal loaded from `data.io.path` after the configured pre-processing steps.
- `general.fs` — float. Sampling frequency $f_s$. Physical units.
- `general.highest_freq` — float.  Upper frequency bound used when defining the smallest admissible scale. Physical units. 
- `general.int_scale` — float. Integral length scale $L$. Physical units.
- `general.taylor_scale` — float. Taylor microscale $\lambda$. Physical units.
- `general.nbins` — int or `"auto"` (=$10\times\text{range}/\sigma$, see `data.stats.range` and `data.stats.std`)

> The *general* `auto` values are computed during the pre-analysis step by **`p0.compute_and_write_general_autovalues`**.

> `auto` values specific to each routine section are calculated by that section’s helper function (e.g. the Markov-related `auto` parameters are set in **`p3.compute_markov_autovalues`**).


## Routine parameters

### Part 1 — Turbulence-analysis <a name="part1"></a>

#### `plot_stationary`
Overlay equal-length chunks of the signal to check stationarity.

- `data_split_percent` — float [0 – 100]. Chunk length as a percentage of the data.


#### `plot_pdf`
Draw the empirical probability-density function.

- `nbins` — int or `"auto"` (= `general.nbins`). Histogram bin count.


#### `plot_spectrum`
Plot both the raw spectrum $E(f)$ and the compensated spectrum $E(f)\,f^{p}$.

- `comp_exponent` — float (string). Exponent $p$ for spectral compensation.  
- `moving_average_nbins` — int, `"auto"` (= $10 \times$ `general.nbins`), or `null`. Log-spaced smoothing windows; `null` disables it.  
- `general.fs` — sets the frequency axis.  
- `general.int_scale` — marks the integral-scale position $1/L$.  
- `general.taylor_scale` — marks the Taylor-microscale position $1/\lambda$.


###  Part 2 — Markov <a name="part2"></a>
#### General
- `p2.general.markov_scale_us` — int or `"auto"` (=$\lambda\,f_s/U$). Sample units.
- `nbins` — int or `"auto"`(=`general.nbins`). Default histogram bin count for all Part 2 PDFs.
- `p2.general.min_events` — int. Minimum per-bin count tolerable.

#### `compute_wilcoxon_test`
Compute the standardised Wilcoxon statistic $W_T(\Delta)$ to evaluate the Markov property of velocity increments.

- `nbins` — int or `"auto"` (= `p2.general.nbins`). Histogram bins for the increment PDFs.
- `indep_scale` — float or `"auto"` (derived from `general.int_scale`). Physical units. Independent scale of the data.
- `end_scale` — float or `"auto"` (= `indep_scale`). Largest lag $\Delta$ considered.
- `n_interv_sec` — int (default 1). Number of intervals considered in each independent scale. Increasing this value gives a softer $W_T(\Delta)$ curve in exchange for longer execution time, it is typically reasonable to try values between $1$ and $5$.
- `general.fs`, `general.taylor_hyp_vel` — Converts physical units to sample units.

#### `plot_wilcoxon_test`
Plot $W_T(\Delta)$ and the conditional PDFs at the Markov scale $\Delta_\text{EM}$.

- `p2.general.markov_scale_us` — Marks the estimated Markov scale $\Delta_{\mathrm{EM}}$ on the plot.

#### `compute_conditional_moments_estimation`
Estimate conditional moments $M_{11},M_{21},M_{31},M_{41}$ and the associated density functions over a logarithmic ladder of scales; writes 
`conditional_moments_estimation.npz` and `density_functions.npz`.

- `n_scale_steps` — int. Number of logarithmically spaced scales between the Taylor and integral scales.
- `p2.general.markov_scale_us` — Sets the small offset $\Delta$ used to build $u_{s+\Delta}-u_s$.
- `p2.general.nbins` — int or `"auto"` (= `general.nbins`). Histogram bins for all PDFs.
- `p2.general.min_events` — Minimum per-bin count required to accept a scale.
- `general.taylor_scale` — Lower limit for the scale ladder.
- `general.highest_freq` — Fixes the smallest admissible scale.
- `general.fs`, `general.taylor_hyp_vel` — Converts physical units to sample units.

#### `compute_km_coeffs_estimation`
Estimate the Kramers–Moyal coefficients $D^{(1)}\!\dots D^{(4)}$ from the previously saved conditional moments; the result is stored in `km_coeffs_estimation.npz`.

- `p2.general.markov_scale_us` — Fixes the short offset $\Delta$ used in all fits.
- `p2.general.nbins` — Histogram bins for the increment PDFs.
- `p2.general.min_events` — Minimum per-bin count required to accept a coefficient.
- `general.taylor_scale` — Normalises the fitting steps.
- `general.fs`, `general.taylor_hyp_vel` — Converts physical units to sample units.

#### `plot_km_coeffs_estimation`
3-D scatter plots of $D^{(1)}\!\dots D^{(4)}$ against the normalised increment $u_s/\sigma_\infty$ and scale $s/\lambda$.

- `general.taylor_scale` — Normalise the scale axis.

#### `compute_km_coeffs_estimation_stp_opti`
Refine the preliminary Kramers–Moyal estimates by minimising the divergence between the empirical PDF and the short-time propagator; writes the optimised coefficients (`D1_opti`, `D2_opti`) back into `km_coeffs_estimation.npz`.

- `tol` — float [0 – 1]. Relative bound for each parameter during the optimization.
- `p2.general.nbins` — Histogram bins for all PDFs.
- `general.taylor_scale` — Normalises the scales before doing the optimization.
- `general.fs`, `general.taylor_hyp_vel` — Converts physical units to sample units.

#### `plot_km_coeffs_estimation_opti`
Compare unoptimised and optimised coefficients in 3-D scatter plots and overlay their short-time PDF reconstructions.

- `p2.general.markov_scale_us` — The scale used to build the PDF is a proportion of $\Delta_\text{EM}$.
- `p2.general.nbins` — Bins used for the PDF reconstruction.
- `general.taylor_scale` — Normalise the scale axis.
- `general.fs`, `general.taylor_hyp_vel` — Converts physical units to sample units.

#### `compute_km_coeffs_fit`
Fit analytical surfaces to the optimised and non-optimised Kramers–Moyal estimates and save the results to `km_coeffs_stp_opti.npz` (short-time–propagator fit) and `km_coeffs_no_opti.npz` (raw fit).

- `p2.general.nbins` — Bins used when rebuilding the data arrays for fitting.
- `general.taylor_scale` — Normalises the scale coordinates in the regression.

#### `plot_km_coeffs_fit`
Display the fitted $D^{(1)}$ and $D^{(2)}$ surfaces together with the scatter of optimized estimates.

- `p2.general.nbins` — Bins used to rebuild the scatter data.
- `general.taylor_scale` — Normalises the scale axis.


### Part 3 — Entropy production <a name="part3"></a>

#### General
- `p3.general.prop_to_use` — either a float float [0 – 1] (for 1-D data) or a two-floats list `[[0 – 1], [0 – 1]]` (for 2-D data). Specifies the fraction(s) of the **already-loaded** dataset.
- `p3.general.smallest_scale` — float or `"auto"` (= `general.taylor_scale`). Minimum cascade scale $s_{\min}$. Physical units.
- `p3.general.largest_scale` — float or `"auto"` (= `general.int_scale`). Maximum cascade scale $s_{\max}$. Physical units.
- `p3.general.scale_subsample_step_us` — int or `"auto"` (= `p2.general.markov_scale_us`). Sub-sampling step between scales. Sample units.
- `p3.general.overlap_trajs_flag` — bool. Enable (`true`) or disable (`false`) overlapping trajectories.
- `p3.general.available_ram_gb` — float. Available RAM for chunking (GB).

#### `compute_km_coeffs_ift_opti`
Refine the short-time–propagator coefficients so that they also satisfy the **integral fluctuation theorem (IFT)**; the resulting set is saved to `km_coeffs_ift_opti.npz`.

It uses `km_coeffs_stp_opti.npz` for the initial coefficients and `km_coeffs_estimation.npz` to get the optimisation scales.

- `tol_D1` — float [0 – 1]. Relative search window for each $d_{11}(s)$ value.  
- `tol_D2` — float [0 – 1]. Relative search window for each $d_{2j}(s)$ value.  
- `iter_max` — int. Maximum L-BFGS-B iterations. The algorithm is very efficient finding the minimum but the general optimization is slow, so ~ 5 is a good number to start.
- `p3.general.smallest_scale` — Minimum cascade scale $s_{\min}$.
- `p3.general.largest_scale` — Maximum cascade scale $s_{\max}$.
- `p3.general.scale_subsample_step_us` — Sub-sampling step between scales.
- `p3.general.overlap_trajs_flag` — Whatever use or not overlapping trajectories.
- `p3.general.available_ram_gb` — Available RAM for chunking (GB). 
- `general.taylor_scale` — Normalises the scales before doing the optimization.
- `general.fs`, `general.taylor_hyp_vel` — Converts physical units to sample units.

#### `compute_entropy_(stp|ift)_opti`
Compute the medium, system and total entropy along the cascade using the (**short-time-propagator–optimized|ift-optimized**) KM coefficients; results are saved to `entropies_(stp|ift)_opti.npz`.

- `p3.general.smallest_scale` — Minimum cascade scale $s_{\min}$.
- `p3.general.largest_scale` — Maximum cascade scale $s_{\max}$.
- `p3.general.scale_subsample_step_us` — Sub-sampling step between scales.
- `p3.general.overlap_trajs_flag` — Whatever use or not overlapping trajectories.
- `p3.general.available_ram_gb` — Available RAM for chunking (GB). 

#### `plot_entropy_and_ift_for_(ift|stp)_opti`
Plot the PDFs of medium, system and total entropy, using the data from `entropies_(stp|ift)_opti.npz`. Check the integral fluctuation theorem (IFT) and the detailed fluctuation theorem (DFT).

- `nbins` — int. Histogram bins for the entropy PDFs and DFT plot.
