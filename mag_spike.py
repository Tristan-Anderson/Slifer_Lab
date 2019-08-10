from slifercal import slifercal as sc

k = sc()
k.plot_magnet_spikes(thermistors=["CCCS.T3", "CX.T2", "CCX.T1", "CCS.F11", "CCS.F10", "CCS.F9"], keywords=["magnet", "tesla", "field", "strength"], kelvin=True)
