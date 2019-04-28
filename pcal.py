
from slifercal import slifercal as sc
k = sc()
k.load_data()
k.plot_calibration_candidates(n_best=3, plot_logbook=True, dpi_val=50, data_record=True)
