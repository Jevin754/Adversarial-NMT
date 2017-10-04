#!/usr/bin/env python
import optparse
import sys
from collections import defaultdict

def ibm1_train(bitext,t_probability):

    # initialize t(f|e) uniformly
    for (n, (f, e)) in enumerate(bitext):
        for f_i in set(f):
            for e_j in set(e):
                t_probability[(f_i, e_j)] = 1.0 / len(f)
    
    k = 0

    while k < 10:
        k += 1
        
        e_count = defaultdict(float)  # count the english word
        fe_count = defaultdict(float)     # count (f,e) word pair

        for (n, (f, e)) in enumerate(bitext):
            
            # computer normalization
            for f_i in set(f):
                Z = 0
                for e_j in set(e):
                    Z += t_probability[(f_i, e_j)]

                for e_j in set(e):
                    cur = 1.0 * t_probability[(f_i, e_j)] / Z
                    fe_count[(f_i, e_j)] += cur
                    e_count[e_j] += cur
                
        # estimate probabilities
        for (f_i, e_j) in fe_count.keys():
            t_probability[(f_i, e_j)] = fe_count[(f_i, e_j)] / e_count[e_j]

    return t_probability

def ibm1_align(bitext,t_probability):
    # align by the largest t
    for (f, e) in bitext:
        for (i, f_i) in enumerate(f):
            max_p = 0.0
            best_j = 0
            for (j, e_j) in enumerate(e):
                if t_probability[(f_i, e_j)] > max_p:
                    max_p = t_probability[(f_i, e_j)]
                    best_j = j
            sys.stdout.write("%i-%i " % (i,best_j))
        sys.stdout.write("\n")

