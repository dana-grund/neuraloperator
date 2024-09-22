from naive_crps import naive_crps
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from naive_estimator.naiveDataset_other import get_stabilitySet, get_processor

"""
Script for computing the CRPS on the stability dataset for the naive estimator.
"""

stability_db = get_stabilitySet()
data_processor = get_processor(stability_db)

numEns = 10

startComp = time.time()
crps = naive_crps(stability_db, data_processor, numEns)
endComp = time.time()

print(f"Stability naive crps ({numEns} ensembles): {crps}")
print(f"Calculation time: {endComp-startComp}")

f = open(f"stabilityNaiveCrpsFNO{numEns}_2.txt", "x")
f.write(f"{crps}")
f.close()