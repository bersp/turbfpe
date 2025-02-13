from turbfpe.part0_preanalysis import preanalysis
from turbfpe.part1_turbulence_analysis import turbulence_analysis
from turbfpe.part2_markov import markov

params_filename = './Renner_params.toml'

preanalysis.exec_rutine(params_filename)
turbulence_analysis.exec_rutine(params_filename)
markov.exec_rutine(params_filename)
