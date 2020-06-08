from slifercal import slifercal as sc

k = sc(datafile_location="data_record_12-20-2019.csv", logbook_datafile_location="logbook.csv")

k.keyword(keywords=['VME', "TE", "VNE"], persistance=True)
