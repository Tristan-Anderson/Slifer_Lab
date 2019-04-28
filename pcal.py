
from slifercal import slifercal as sc
k = sc()
k.load_data()
k.plot_calibration_candidates(n_best=15, plot_logbook=True, dpi_val=250, data_record=True)
