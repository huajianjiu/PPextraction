import numpy as np
import os.path
import cPickle as pickle
import sys


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

    def save_ppdb(self, pkl=True, text=True):
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
        self.vocab = vocab
        self.ppdb_paraphrases = ppdb_paraphrases = {}
        with open(vocab, "r") as f_vocab:
                words = f_vocab.readlines()
                words = [x.split()[0] for x in words]
                self.words = words
        with open(ppdb, "r") as ppdb_f:
            lines = ppdb_f.readlines()
            print "Total lines: " + str(len(lines))
            n = 0
            for line in lines:
                if (line.split("|||")[1].strip() in words) and (line.split("|||")[2].strip() in words):
                    baseword = line.split("|||")[1].strip()
                    ppword = line.split("|||")[2].strip()
                    if (line.split("|||")[-1].strip() == "Equivalence"):
                        self.add_paraphrases(baseword, ppword)
                        self.add_paraphrases(ppword, baseword)
                    elif (line.split("|||")[-1].strip() == "ForwardEntailment"):
                        self.add_paraphrases(baseword, ppword)
                    elif (line.split("|||")[-1].strip() == "ReverseEntailment"):
                        self.add_paraphrases(ppword, baseword)
                n += 1
                if n%10000 == 0:
                    print str(n) + " lines processed."
        print "Finish. Totally "+str(n)+" lines processed."

    def search_baseword(self, inputword):
        return inputword in self.ppdb_paraphrases.keys()
    
    def add_paraphrases(self, baseword, ppword):
        if self.search_baseword(baseword):
            self.ppdb_paraphrases[baseword].append(ppword)
        else:
            self.ppdb_paraphrases[baseword] = [ppword]

    def save_ppdb(self):
        with open("ppdb_2_"+self.vocab+".txt", "w") as f_save:
            n = 0
            for word in self.words:
                if word == "UNK":
                    write_line = "</s> </s>\n"
                elif word in self.ppdb_paraphrases.keys():
                    write_line = str(word) + " "
                    for ppword in self.ppdb_paraphrases[word]:
                        write_line += ppword + " "
                    write_line += "</s>\n"
                else:
                    write_line = str(word) + " </s>\n"
                f_save.write(write_line)
                n += 1
                if n%1000 == 0:
                    f_save.flush()


if __name__ == "__main__":
    if len(sys.argv)>1:
        ppdb_s_corpus = PPDB_2("vocab.txt", sys.argv[1])
    else:
        ppdb_s_corpus = PPDB_2()
    ppdb_s_corpus.save_ppdb()
