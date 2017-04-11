import numpy as np
import os.path
import sys
from nltk.corpus import wordnet as wn


class WNDB(object):
    def __init__(self, vocab="vocab.txt"):
        self.vocab = vocab
        self.paraphrases = {}
        with open(vocab, "r") as f_vocab:
            words = f_vocab.readlines()
            words = [x.split()[0] for x in words]
            self.words = words

    def read_lexicon(self):
        print "Abstract."

    def search_baseword(self, inputword):
        return inputword in self.paraphrases.keys()

    def add_paraphrases(self, baseword, ppword, score):
        if baseword == ppword:
            return
        if self.search_baseword(baseword):
            if ppword in self.paraphrases[baseword]:
                return
            self.paraphrases[baseword] += [ppword, score]
        else:
            self.paraphrases[baseword] = [ppword, score]

    def save_ppdb(self, filename):
        with open(filename, "w") as f_save:
            n = 0
            for word in self.words:
                if word == "UNK":
                    write_line = "</s> </s>\n"
                elif word in self.paraphrases.keys():
                    write_line = str(word) + " "
                    for ppword in self.paraphrases[word]:
                        write_line += str(ppword) + " "
                    write_line += "</s>\n"
                else:
                    write_line = str(word) + " </s>\n"
                f_save.write(write_line)
                n += 1
                if n % 1000 == 0:
                    f_save.flush()


class WNDB_hypo(WNDB):

    def __init__(self, vocab="vocab.txt"):
        super(WNDB_hypo, self).__init__(vocab)

    def read_lexicon(self):
        print "read Wordnet"
        words = self.words
        for word in words:
            synsets=wn.synsets(word)
            for synset in synsets:
                hyponyms = synset.hyponyms()
                for hyponym in hyponyms:
                    tailword = hyponym.name().split(".")[0]
                    score = synset.path_similarity(hyponym)
                    self.add_paraphrases(word, tailword, score)
        print "Finish."

    def save_ppdb(self):
        super(WNDB_hypo, self).save_ppdb("wordnetpp_hypo")


class WNDB_synset(WNDB):
    def __init__(self, vocab="vocab.txt"):
        super(WNDB_synset, self).__init__(vocab)

    def read_lexicon(self):
        print "read Wordnet"
        words = self.words
        for word in words:
            synsets = wn.synsets(word)
            for synset in synsets:
                tailword = synset.name().split(".")[0]
                score = synset.path_similarity(synset)
                self.add_paraphrases(word, tailword, score)
        print "Finish."

    def save_ppdb(self):
        super(WNDB_synset, self).save_ppdb("wordnetpp_synset")



if __name__ == "__main__":
    if len(sys.argv)>1:
        vocab = sys.argv[1]
    else:
        vocab = "vocab.txt"
    wndb = WNDB_synset(vocab)
    wndb.read_lexicon()
    wndb.save_ppdb()
