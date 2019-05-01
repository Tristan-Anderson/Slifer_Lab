from slifercal import slifercal as sc
import time

start = time.time()
k = sc()
k.complete(True)
finish = time.time()
print("Total:",finish-start)