from naive_crps import naiveBaseline_crps
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from naive_estimator.naiveDataset_other import get_baselineSet, get_processor

"""
Script for computing the baseline CRPS for the naive estimator.
"""

set_size = 10000
baseline_db = get_baselineSet(set_size)
data_processor = get_processor(baseline_db)

startComp = time.time()
crps = naiveBaseline_crps(baseline_db, data_processor, set_size)
endComp = time.time()

print(f"Baseline naive crps ({set_size} samples): {crps}")
print(f"Calculation time: {endComp-startComp}")

f = open(f"baselineNaiveCrpsFNO{set_size}_2.txt", "x")
f.write(f"{crps}")
f.close()