
"""
Compute relative probabilistic scores (CRPS and MMD for all 3 datasets) for the naive estimater.
Relies on precomputed absolute scores.
"""

# open file for results
out = open(f"relativeProbScores_correctedEns_new.txt", "x")

# crps
out.write("Crps:\n")
#baseline
#baseCrpsFNO = 0.08613201019701766
baseCrpsFNO = 0.10228725340712341
npt = open("/cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/naive_estimator/crps/baselineNaiveCrpsFNO10000_2.txt", "r")
baseCrpsNaive = float(npt.readline())
baseCrpsRel = baseCrpsFNO / baseCrpsNaive
#control prints
print(f"Base FNO crps: {baseCrpsFNO}")
print(f"Base Naive crps: {baseCrpsNaive}")
print(f"Base Relative crps: {baseCrpsRel}\n")
#write to out
out.write(f"{baseCrpsRel}\n")

#ensemble
#ensCrpsFNO = 0.212027
#ensCrpsFNO = 0.144403
ensCrpsFNO = 0.00969411
npt = open("/cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/naive_estimator/crps/ensembleNaiveCrpsFNO10_corrected2.txt", "r")
ensCrpsNaive = float(npt.readline())
ensCrpsRel = ensCrpsFNO / ensCrpsNaive
#control prints
print(f"Ens FNO crps: {ensCrpsFNO}")
print(f"Ens Naive crps: {ensCrpsNaive}")
print(f"Ens Relative crps: {ensCrpsRel}\n")
#write to out
out.write(f"{ensCrpsRel}\n")

#stability
#stabCrpsFNO = 0.0457168
stabCrpsFNO = 0.0142939
npt = open("/cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/naive_estimator/crps/stabilityNaiveCrpsFNO10_2.txt", "r")
stabCrpsNaive = float(npt.readline())
stabCrpsRel = stabCrpsFNO / stabCrpsNaive
#control prints
print(f"Stab FNO crps: {stabCrpsFNO}")
print(f"Stab Naive crps: {stabCrpsNaive}")
print(f"Stab Relative crps: {stabCrpsRel}\n")
#write to out
out.write(f"{stabCrpsRel}\n")

# mmd
out.write("\nMMD:\n")
#baseline
#baseMMDFNO = 0.2321419063210487
baseMMDFNO = 0.22873455181717875
npt = open("/cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/naive_estimator/mmd/baselineNaiveMmdFNO10000_newSigma.txt", "r")
baseMMDNaive = float(npt.readline())
baseMMDRel = baseMMDFNO / baseMMDNaive
#control prints
print(f"Base FNO mmd: {baseMMDFNO}")
print(f"Base Naive mmd: {baseMMDNaive}")
print(f"Base Relative mmd: {baseMMDRel}\n")
#write to out
out.write(f"{baseMMDRel}\n")

#ensemble
#ensMMDFNO = 0.759707
#ensMMDFNO = 0.892879
ensMMDFNO = 0.60768
npt = open("/cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/naive_estimator/mmd/ensembleNaiveMmdFNO10_correctedNewSigma.txt", "r")
ensMMDNaive = float(npt.readline())
ensMMDRel = ensMMDFNO / ensMMDNaive
#control prints
print(f"Ens FNO mmd: {ensMMDFNO}")
print(f"Ens Naive mmd: {ensMMDNaive}")
print(f"Ens Relative mmd: {ensMMDRel}\n")
#write to out
out.write(f"{ensMMDRel}\n")

#stability
stabMMDFNO = 0.770371
stabMMDFNO = 0.762361
npt = open("/cluster/home/fstuehlinger/ba/git/dana-grund/neuraloperator/naive_estimator/mmd/stabilityNaiveMmdFNO10_newSigma.txt", "r")
stabMMDNaive = float(npt.readline())
stabMMDRel = stabMMDFNO / stabMMDNaive
#control prints
print(f"Stab FNO mmd: {stabMMDFNO}")
print(f"Stab Naive mmd: {stabMMDNaive}")
print(f"Stab Relative mmd: {stabMMDRel}\n")
#write to out
out.write(f"{stabMMDRel}\n")