from naive_mmd import naiveBaseline_mmd
import time
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from naive_estimator.naiveDataset_other import get_baselineSet, get_processor

"""
Script for computing the baseline MMD for the naive estimator.
"""

set_size = 10000
baseline_db = get_baselineSet(set_size)
data_processor = get_processor(baseline_db)

startComp = time.time()
mmd = naiveBaseline_mmd(baseline_db, data_processor, set_size)
endComp = time.time()

print(f"Baseline naive mmd ({set_size}): {mmd}")
print(f"Calculation time: {endComp-startComp}")

f = open(f"baselineNaiveMmdFNO{set_size}_newSigma.txt", "x")
f.write(f"{mmd}")
f.close()