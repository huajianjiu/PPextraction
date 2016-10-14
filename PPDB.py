import numpy as np
import os.path
import cPickle as pickle


class PPDBs4vocab(object):
    # read paraphrases of the words in a specified vocabulary
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
            self.paraphrases_words=ppdb_paraphrases
            self.paraphrases = np.zeros((len(self.words), relations_num), dtype=np.int32)
            for i, word in enumerate(words):
                if word in ppdb_paraphrases.keys():
                    self.paraphrases[i][0] = self.words.index(ppdb_paraphrases[word])

    def save_wordnet(self, pkl=True, text=True):
        if pkl:
            with open("ppdb_s_"+self.vocab+str(self.relations_num)+".pkl", "wb") as f_save:
                pickle.dump(self.words, f_save)
                pickle.dump(self.paraphrases, f_save)
        if text:
            with open("ppdb_s_"+self.vocab+str(self.relations_num)+".txt", "w") as f_save:
                for word in self.words:
                    if word == "UNK":
                        write_line = "</s> </s>\n"
                    elif word in self.paraphrases_words.keys():
                        write_line = str(word) + " " + str(self.paraphrases_words[word]) + "\n"
                    else:
                        write_line = str(word) + " </s>\n"
                    f_save.write(write_line)

class PPDB_2(object):
    def __init__(self, vocab="vocab.txt", ppdb="ppdb-2.0-tldr"):
        pass
    def save_ppdb(slef, pkl=False, text=True):
        pass

if __name__ == "__main__":
    ppdb_s_corpus = PPDBs4vocab()
    ppdb_s_corpus.save_wordnet(True, True)
    for i in range(100):
        print str(ppdb_s_corpus.words[i])+":"+str(ppdb_s_corpus.paraphrases[i])

