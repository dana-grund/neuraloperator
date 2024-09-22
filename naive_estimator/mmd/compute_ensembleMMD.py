from naive_mmd import naive_mmd
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from naive_estimator.naiveDataset_other import get_ensembleSet, get_processor

"""
Script for computing the MMD on the ensemble dataset for the naive estimator.
"""

ensemble_db = get_ensembleSet()
data_processor = get_processor(ensemble_db)

numEns = 10

startComp = time.time()
mmd = naive_mmd(ensemble_db, data_processor, numEns)
endComp = time.time()

print(f"Ensemble naive mmd ({numEns} ensembles): {mmd}")
print(f"Calculation time: {endComp-startComp}")

f = open(f"ensembleNaiveMmdFNO{numEns}_correctedNewSigma.txt", "x")
f.write(f"{mmd}")
f.close()