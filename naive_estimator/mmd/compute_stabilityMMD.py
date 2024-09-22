from naive_mmd import naive_mmd
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from naive_estimator.naiveDataset_other import get_stabilitySet, get_processor

"""
Script for computing the MMD on the stability dataset for the naive estimator.
"""

stability_db = get_stabilitySet()
data_processor = get_processor(stability_db)

numEns = 10

startComp = time.time()
mmd = naive_mmd(stability_db, data_processor, numEns)
endComp = time.time()

print(f"Stability naive mmd ({numEns} ensembles): {mmd}")
print(f"Calculation time: {endComp-startComp}")

f = open(f"stabilityNaiveMmdFNO{numEns}_newSigma.txt", "x")
f.write(f"{mmd}")
f.close()