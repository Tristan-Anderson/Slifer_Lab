from slifercal import slifercal as sc
import time

start = time.time()
k = sc(datafile_location="smol_data.csv")
k.complete(True)
finish = time.time()
print("Total:",finish-start)
