import optparse
import sys
from collections import defaultdict
from ibm1f2e import *
from ibm2 import *


from nltk.stem import SnowballStemmer
reload(sys)
sys.setdefaultencoding("latin-1") # required for stemming

def main():
    optparser = optparse.OptionParser()
    optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
    optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
    optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
    optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint, type="int", help="Number of sentences to use for training and alignment")
    (opts, _) = optparser.parse_args()
    f_data = "%s.%s" % (opts.train, opts.french)
    e_data = "%s.%s" % (opts.train, opts.english)

    t_probability = defaultdict(float) # lexical translation probability

    bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))[:opts.num_sents]]
    
    # Stem words before model training
    french_stemmer = SnowballStemmer("french")
    english_stemmer = SnowballStemmer("english")
    bitext_stemmed = []
    for (n, (f, e)) in enumerate(bitext):
        f_stemmed = [french_stemmer.stem(word.decode("utf-8")) for word in f]
        e_stemmed = [english_stemmer.stem(word) for word in e]
        bitext_stemmed.append([f_stemmed, e_stemmed])

    bitext = bitext_stemmed

    # Train the model and test
    t_prob_tab1 = ibm1f2e_train(bitext,t_probability)

    t_prob_tab2, align_prob = ibm2_train(bitext,t_prob_tab1)
    ibm2_align(bitext,t_prob_tab2, align_prob)

if __name__ == "__main__":
    main()
