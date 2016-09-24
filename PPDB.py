import numpy as np
import os.path
import cPickle as pickle


class PPDB_s(object):
    def __init__(self, relations_num=1, vocab="vocab.txt", ppdb="ppdb-1.0-s-lexical"):
        self.relations_num = relations_num
        self.vocab = vocab
        if os.path.isfile("ppdb_s_" + self.vocab + str(self.relations_num) + ".pkl"):
            with open("ppdb_s_" + self.vocab + str(self.relations_num) + ".pkl", "rb") as f_load:
                self.words = pickle.load(f_load)
                self.paraphrases = pickle.load(f_load)
        else:
            with open(vocab, "r") as f_vocab:
                words = f_vocab.readlines()
                words = [x.split()[0] for x in words]
                self.words = words
            with open(ppdb, "r") as ppdb_f:
                ppdb_paraphrases = {line.split("|||")[1].strip(): line.split("|||")[2].strip()
                                    for line in ppdb_f.readlines() if (line.split("|||")[1].strip() in words)
                                    and (line.split("|||")[2].strip() in words)}
            self.paraphrases = np.zeros((len(self.words), relations_num), dtype=np.int32)
            for i, word in enumerate(words):
                if word in ppdb_paraphrases.keys():
                    self.paraphrases[i][0] = self.words.index(ppdb_paraphrases[word])
            self.save_wordnet()

    def save_wordnet(self):
        with open("ppdb_s_"+self.vocab+str(self.relations_num)+".pkl", "wb") as f_save:
            pickle.dump(self.words, f_save)
            pickle.dump(self.paraphrases, f_save)

if __name__ == "__main__":
    wordnet_net = PPDB_s()
    for i in range(100):
        print str(wordnet_net.words[i+100])+":"+str(wordnet_net.paraphrases[i+100])

