from turbfpe.part0_preanalysis import preanalysis
from turbfpe.part1_turbulence_analysis import turbulence_analysis
from turbfpe.part2_markov import markov
from turbfpe.part3_entropy import entropy

params_filename = './Renner_params.toml'

preanalysis.exec_rouine(params_filename)
turbulence_analysis.exec_rouine(params_filename)
markov.exec_rouine(params_filename)
entropy.exec_rouine(params_filename)
