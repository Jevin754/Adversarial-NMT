# MT668
Fall 2017 Machine Translation

Homework 2(Word Alignment): We implemented IBM Model 1 and IBM Model 2.

How to run IBM Model 1:

python align1.py -n 1000 > ibm1.a

the default model is f2e. if you want to run e2f,

python align1.py -n 1000 -m e2f > ibm1.a

How to run IBM Model 2:

python align2.py -n 1000 > ibm2.a


ibm1.a and ibm2.a are the result of 10W sentence of Model 1 and 2

ibm1_output.txt and ibm2.output.txt are the first 1000 lines of result
