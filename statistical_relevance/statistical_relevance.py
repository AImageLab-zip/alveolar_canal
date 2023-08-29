import pandas as pd
import numpy as np
from scipy import stats
import scipy.special as special

# STO TTEST REL NON MI PIACE COME CALCOLA I DOF

def vitto_ttest(a, b, alternative="two-sided", verbose=False):
    if verbose:
        print("\nalternative: ", alternative)
    a = np.array(a)
    b = np.array(b)

    a_m = a.mean()
    a_v = a.var(ddof=1)
    a_l = a.shape[0]

    b_m = b.mean()
    b_v = b.var(ddof=1)
    b_l = b.shape[0]

    t = (a_m-b_m)/np.sqrt(((a_v/a_l)+(b_v/b_l)))
    if verbose:
        print("t: ", t)

    numerator   = (((a_v/a_l)+(b_v/b_l))**2)
    denominator = ((((a_v/a_l)**2)/(a_l-1))+(((b_v/b_l)**2)/(b_l-1)))
    v = numerator/denominator
    v = int(v) # v must be rounded down to the nearest integer
    if verbose:
        print("v: ", v)
    if alternative=="less":
        pval = special.stdtr(v, t)
    elif alternative=="greater":
        pval = special.stdtr(v, -t)
    elif alternative=="two-sided":
        pval = special.stdtr(v, -np.abs(t))*2
    else:
        raise ValueError("alternative must be 'less', 'greater' or 'two-sided'")
    if verbose:
        print("pval: ", pval)
    return pval

experiments_dict = {
    'baseline'      :[0.8017, 0.7921, 0.8031, 0.7966, 0.7921, 0.7944, 0.7848, 0.7979, 0.7995, 0.8029],
    'TL4H4_EUH'     :[0.8062, 0.8043, 0.8115, 0.7969, 0.7865, 0.7962, 0.7876, 0.7906, 0.7942, 0.7874],
    'ATL4H4_EUH'    :[0.8063, 0.7989, 0.7987, 0.8010, 0.8026, 0.7977, 0.7916, 0.8023, 0.8105, 0.7957],
    'MATL4H4_EUH'   :[0.8078, 0.8046, 0.8018, 0.8025, 0.8009, 0.8033, 0.8008, 0.8093, 0.7960, 0.7955],
}

experiments_df = pd.DataFrame(experiments_dict)

def ttest(a, b):
    result = stats.ttest_ind(a, b, alternative='less')
    return result.pvalue

def ttest_passed(a, b, alpha=0.05):
    result = stats.ttest_ind(a, b, alternative='less')
    return result.pvalue<alpha

print("\nGeneral statistics:\n", experiments_df.describe())

ttest_dict = {
    "first measure" : [ "baseline",
                        "baseline"],
    "second measure": [ "ATL4H4_EUH", 
                        "MATL4H4_EUH"],
    "p-value"       : [ ttest(experiments_dict["baseline"], experiments_dict["ATL4H4_EUH"]), 
                        ttest(experiments_dict["baseline"], experiments_dict["MATL4H4_EUH"])],
    # "alpha"         : ["0.10", "0.05"],
    # "H0 rejected"   : [ttest_passed(experiments_dict["baseline"], experiments_dict["ATL4H4_EUH"], alpha=0.1), ttest_passed(experiments_dict["baseline"], experiments_dict["MATL4H4_EUH"])],
}
ttest_df = pd.DataFrame(ttest_dict)

print("\nPaired Samples t-test:\n", ttest_df)

vitto_ttest(experiments_dict["baseline"], experiments_dict["ATL4H4_EUH"], alternative="less", verbose=True)
vitto_ttest(experiments_dict["baseline"], experiments_dict["MATL4H4_EUH"], alternative="less", verbose=True)

# Devore example
No_fusion  = [2748, 2700, 2655, 2822, 2511, 3149, 3257, 3213, 3220, 2753]
Fused      = [3027, 3356, 3359, 3297, 3125, 2910, 2889, 2902]
vitto_ttest(No_fusion, Fused, alternative="less", verbose=True) # T = 1.80 P = 0.046 DF = 15

