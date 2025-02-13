from turbfpe.part0_preanalysis import preanalysis
from turbfpe.part1_turbulence_analysis import turbulence_analysis

params_filename = './params_Renner.toml'

preanalysis.exec_rutine(params_filename)
turbulence_analysis.exec_rutine(params_filename)
