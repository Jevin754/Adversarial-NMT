import optparse
import sys
from collections import defaultdict


def ibm2_train(bitext,t_probability):

    # Initialize align probability
    a = defaultdict(float)
    for (f, e) in bitext:
        lf = len(f)
        le = len(e)
        for j, e_j in enumerate(e):
            for i, f_i in enumerate(f):
                a[(j, i, le, lf)] = 1.0 / (lf + 1)

    # Main loop
    k = 0
    while k < 5:
        k += 1
        count_e_given_f = defaultdict(float)
        total_f = defaultdict(float)
        count_align = defaultdict(float)
        total_align = defaultdict(float)
        s_total = defaultdict(float)

        for (f, e) in bitext:

            le = len(e)
            lf = len(f)

            # Compute Normalization
            for (j, e_j) in enumerate(e):
                s_total[e_j] = 0.0
                for (i, f_i) in enumerate(f):
                    s_total[e_j] += t_probability[(e_j, f_i)] * a[(j,i,le,lf)]

            # Collect counts
            for (j, e_j) in enumerate(e):
                for (i, f_i) in enumerate(f):
                    c = (t_probability[(e_j, f_i)] * a[(j,i,le,lf)]) * 1.0 / s_total[e_j] * 1.0
                    count_e_given_f[(e_j, f_i)] += c
                    total_f[f_i] += c
                    count_align[(j,i,le,lf)] += c
                    total_align[(j,le,lf)] += c

        # Estimate probabilities
        for (e_j, f_i) in t_probability.keys():
            t_probability[(e_j, f_i)] = count_e_given_f[(e_j, f_i)] / total_f[f_i] 

        for (f,e) in bitext:
            le = len(e)
            lf = len(f)
            for (j,e_j) in enumerate(e):
              for (i,f_i) in enumerate(f):
                a[(j,i,le,lf)] = count_align[(j,i,le,lf)] / total_align[(j,le,lf)]



    return t_probability, a

def ibm2_align(bitext,t_probability, a):
    # align by the largest t
    for (f, e) in bitext:
        le = len(e)
        lf = len(f)
        for (j, e_j) in enumerate(e):
            best_p = 0
            best_i = 0
            for (i, f_i) in enumerate(f):
                cur = t_probability[(e_j, f_i)] * a[(j,i,le,lf)]
                if cur > best_p:
                    best_p = cur
                    best_i = i
            sys.stdout.write("%i-%i " % (best_i, j))
        sys.stdout.write("\n")






