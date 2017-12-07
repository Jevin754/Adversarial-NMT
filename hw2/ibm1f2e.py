import optparse
import sys
from collections import defaultdict


def ibm1f2e_train(bitext,t_probability):

    # initialize t(e|f) uniformly
    for(n,(f,e)) in enumerate(bitext):
        for e_j in set(e):
            for f_i in set(f):
                t_probability[(e_j,f_i)] = 1.0 / len(f)     

    k = 0

    while k < 10:
        k += 1
        
        ef_count = defaultdict(float)
        f_count = defaultdict(float)
        # computer normalization
        for (n,(f,e)) in enumerate(bitext):
        
            for e_j in set(e):
                Z = 0
                for f_i in set(f):
                    Z += t_probability[(e_j,f_i)]
                # collect counts
                for f_i in set(f):
                    c = 1.0 * t_probability[(e_j,f_i)] / Z
                    ef_count[(e_j,f_i)] += c
                    f_count[f_i] += c

        # estimate probabilities
        for (e_j,f_i) in ef_count.keys():
            t_probability[(e_j,f_i)] = ef_count[(e_j,f_i)] / f_count[f_i]

    return t_probability


def ibm1f2e_align(bitext,t_probability):
    # align by the largest t
    for (f, e) in bitext:
      for (j, e_j) in enumerate(e):
        max_p = 0
        best_i = 0
        for (i, f_i) in enumerate(f):
          if t_probability[(e_j,f_i)] >= max_p:
            max_p = t_probability[(e_j,f_i)]
            best_i = i
        sys.stdout.write("%i-%i " % (best_i,j))
      sys.stdout.write("\n")
