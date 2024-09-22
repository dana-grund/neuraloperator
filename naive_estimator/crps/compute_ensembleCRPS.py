from naive_crps import naive_crps
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from naive_estimator.naiveDataset_other import get_ensembleSet, get_processor

"""
Script for computing the CRPS on the ensemble dataset for the naive estimator.
"""

ensemble_db = get_ensembleSet()
data_processor = get_processor(ensemble_db)

numEns = 10

startComp = time.time()
crps = naive_crps(ensemble_db, data_processor, numEns)
endComp = time.time()

print(f"Ensemble naive crps ({numEns} ensembles): {crps}")
print(f"Calculation time: {endComp-startComp}")

f = open(f"ensembleNaiveCrpsFNO{numEns}_corrected3.txt", "x")
f.write(f"{crps}")
f.close()