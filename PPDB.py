import numpy as np
import os.path
import cPickle as pickle
import sys


# class PPDBs4vocab(object):
#     # read paraphrases of the words in a specified vocabulary
#     def __init__(self, relations_num=5, vocab="vocab.txt", ppdb="ppdb-1.0-s-lexical"):
#         self.relations_num = relations_num
#         self.vocab = vocab
#         if os.path.isfile("ppdb_s_" + self.vocab + str(self.relations_num) + ".pkl"):
#             with open("ppdb_s_" + self.vocab + str(self.relations_num) + ".pkl", "rb") as f_load:
#                 self.words = pickle.load(f_load)
#                 self.paraphrases = pickle.load(f_load)
#         else:
#             with open(vocab, "r") as f_vocab:
#                 words = f_vocab.readlines()
#                 words = [x.split()[0] for x in words]
#                 self.words = words
#             with open(ppdb, "r") as ppdb_f:
#                 ppdb_paraphrases = {line.split("|||")[1].strip(): line.split("|||")[2].strip()
#                                     for line in ppdb_f.readlines() if (line.split("|||")[1].strip() in words)
#                                     and (line.split("|||")[2].strip() in words)}
#             self.paraphrases_words=ppdb_paraphrases
#             self.paraphrases = np.zeros((len(self.words), relations_num), dtype=np.int32)
#             for i, word in enumerate(words):
#                 if word in ppdb_paraphrases.keys():
#                     self.paraphrases[i][0] = self.words.index(ppdb_paraphrases[word])
#
#     def save_ppdb(self, pkl=True, text=True):
#         if pkl:
#             with open("ppdb_s_"+self.vocab+str(self.relations_num)+".pkl", "wb") as f_save:
#                 pickle.dump(self.words, f_save)
#                 pickle.dump(self.paraphrases, f_save)
#         if text:
#             with open("ppdb_s_"+self.vocab+str(self.relations_num)+".txt", "w") as f_save:
#                 for word in self.words:
#                     if word == "UNK":
#                         write_line = "</s> </s>\n"
#                     elif word in self.paraphrases_words.keys():
#                         write_line = str(word) + " " + str(self.paraphrases_words[word]) + "\n"
#                     else:
#                         write_line = str(word) + " </s>\n"
#                     f_save.write(write_line)

class PPDB_2(object):
    def __init__(self, vocab="vocab.txt", ppdb="ppdb-2.0-tldr"):
        self.vocab = vocab
        self.ppdb = ppdb
        self.ppdb_paraphrases = ppdb_paraphrases = {}
        with open(vocab, "r") as f_vocab:
                words = f_vocab.readlines()
                words = [x.split()[0] for x in words]
                self.words = words
    
    def read_lexicon(self):
        print "Abstract Function"

    def search_baseword(self, inputword):
        return inputword in self.ppdb_paraphrases.keys()
    
    def add_paraphrases(self, baseword, ppword, score):
        if baseword == ppword:
            return
        if self.search_baseword(baseword):
            if ppword in self.ppdb_paraphrases[baseword]:
                return
            self.ppdb_paraphrases[baseword] += [ppword, score]
        else:
            self.ppdb_paraphrases[baseword] = [ppword, score]

    def save_ppdb(self, filename):
        with open(filename, "w") as f_save:
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

class PPDB2E(PPDB_2):
    def __init__(self, vocab="vocab.txt", ppdb="ppdb-2.0-tldr"):
        super(PPDB2E, self).__init__(vocab, ppdb)

    def read_lexicon(self):
        print "read Equivalence"
        words = self.words
        ppdb = self.ppdb
        with open(ppdb, "r") as ppdb_f:
            lines = ppdb_f.readlines()
            print "Total lines: " + str(len(lines))
            n = 0
            for line in lines:
                if (line.split("|||")[1].strip() in words) and (line.split("|||")[2].strip() in words):
                    baseword = line.split("|||")[1].strip()
                    ppword = line.split("|||")[2].strip()
                    score = line.split("|||")[3].split(" ")[1].split("=")[1]
                    if (line.split("|||")[-1].strip() == "Equivalence"):
                        self.add_paraphrases(baseword, ppword, score)
                        self.add_paraphrases(ppword, baseword, score)
                n += 1
                if n%10000 == 0:
                    print str(n) + " lines processed."
        print "Finish. Totally "+str(n)+" lines processed."
    
    def save_ppdb(self):
        "write ppdb2e"
        super(PPDB2E, self).save_ppdb("ppdb_2_E.txt")

class PPDB2FR(PPDB_2):
    def __init__(self, vocab="vocab.txt", ppdb="ppdb-2.0-tldr"):
        super(PPDB2FR, self).__init__(vocab, ppdb)

    def read_lexicon(self):
        print "read Forward Reverse"
        words = self.words
        ppdb = self.ppdb
        with open(ppdb, "r") as ppdb_f:
            lines = ppdb_f.readlines()
            print "Total lines: " + str(len(lines))
            n = 0
            for line in lines:
                if (line.split("|||")[1].strip() in words) and (line.split("|||")[2].strip() in words):
                    baseword = line.split("|||")[1].strip()
                    ppword = line.split("|||")[2].strip()
                    score = line.split("|||")[3].split(" ")[1].split("=")[1]
                    if (line.split("|||")[-1].strip() == "ForwardEntailment"):
                        self.add_paraphrases(baseword, ppword, score)
                    elif (line.split("|||")[-1].strip() == "ReverseEntailment"):
                        self.add_paraphrases(ppword, baseword, score)
                n += 1
                if n%10000 == 0:
                    print str(n) + " lines processed."
        print "Finish. Totally "+str(n)+" lines processed."

    def save_ppdb(self):
        "write ppdb2fr"
        super(PPDB2FR, self).save_ppdb("ppdb_2_FR.txt")

class PPDB2EFR(PPDB_2):
    def __init__(self, vocab="vocab.txt", ppdb="ppdb-2.0-tldr"):
        super(PPDB2EFR, self).__init__(vocab, ppdb)

    def read_lexicon(self):
        print "read Equivalence Forward Reverse"
        words = self.words
        ppdb = self.ppdb
        with open(ppdb, "r") as ppdb_f:
            lines = ppdb_f.readlines()
            print "Total lines: " + str(len(lines))
            n = 0
            for line in lines:
                if (line.split("|||")[1].strip() in words) and (line.split("|||")[2].strip() in words):
                    baseword = line.split("|||")[1].strip()
                    ppword = line.split("|||")[2].strip()
                    score = line.split("|||")[3].split(" ")[1].split("=")[1]
                    if (line.split("|||")[-1].strip() == "Equivalence"):
                        self.add_paraphrases(baseword, ppword, score)
                        self.add_paraphrases(ppword, baseword, score)
                    elif (line.split("|||")[-1].strip() == "ForwardEntailment"):
                        self.add_paraphrases(baseword, ppword, score)
                    elif (line.split("|||")[-1].strip() == "ReverseEntailment"):
                        self.add_paraphrases(ppword, baseword, score)
                n += 1
                if n%10000 == 0:
                    print str(n) + " lines processed."
        print "Finish. Totally "+str(n)+" lines processed."

    def save_ppdb(self):
        "write ppdb2efr"
        super(PPDB2EFR, self).save_ppdb("ppdb_2_EFR.txt")

class PPDB2EFRO(PPDB_2):
    def __init__(self, vocab="vocab.txt", ppdb="ppdb-2.0-tldr"):
        super(PPDB2EFRO, self).__init__(vocab, ppdb)

    def read_lexicon(self):
        print "read Equivalence Forward Reverse OtherRelated"
        words = self.words
        ppdb = self.ppdb
        with open(ppdb, "r") as ppdb_f:
            lines = ppdb_f.readlines()
            print "Total lines: " + str(len(lines))
            n = 0
            for line in lines:
                if (line.split("|||")[1].strip() in words) and (line.split("|||")[2].strip() in words):
                    baseword = line.split("|||")[1].strip()
                    ppword = line.split("|||")[2].strip()
                    score = line.split("|||")[3].split(" ")[1].split("=")[1]
                    if (line.split("|||")[-1].strip() == "Equivalence"):
                        self.add_paraphrases(baseword, ppword, score)
                        self.add_paraphrases(ppword, baseword, score)
                    elif (line.split("|||")[-1].strip() == "ForwardEntailment"):
                        self.add_paraphrases(baseword, ppword, score)
                    elif (line.split("|||")[-1].strip() == "ReverseEntailment"):
                        self.add_paraphrases(ppword, baseword, score)
                    elif (line.split("|||")[-1].strip() == "OtherRelated"):
                        self.add_paraphrases(ppword, baseword, score)
                        self.add_paraphrases(baseword, ppword, score)
                n += 1
                if n%10000 == 0:
                    print str(n) + " lines processed."
        print "Finish. Totally "+str(n)+" lines processed."

    def save_ppdb(self):
        "write ppdb2efro"
        super(PPDB2EFRO, self).save_ppdb("ppdb_2_EFRO.txt")

class PPDB2EFROX(PPDB_2):
    def __init__(self, vocab="vocab.txt", ppdb="ppdb-2.0-tldr"):
        super(PPDB2EFROX, self).__init__(vocab, ppdb)

    def read_lexicon(self):
        print "read Equivalence Forward Reverse OtherRelated Exclusion"
        words = self.words
        ppdb = self.ppdb
        with open(ppdb, "r") as ppdb_f:
            lines = ppdb_f.readlines()
            print "Total lines: " + str(len(lines))
            n = 0
            for line in lines:
                if (line.split("|||")[1].strip() in words) and (line.split("|||")[2].strip() in words):
                    baseword = line.split("|||")[1].strip()
                    ppword = line.split("|||")[2].strip()
                    score = line.split("|||")[3].split(" ")[1].split("=")[1]
                    if (line.split("|||")[-1].strip() == "Equivalence"):
                        self.add_paraphrases(baseword, ppword, score)
                        self.add_paraphrases(ppword, baseword, score)
                    elif (line.split("|||")[-1].strip() == "ForwardEntailment"):
                        self.add_paraphrases(baseword, ppword, score)
                    elif (line.split("|||")[-1].strip() == "ReverseEntailment"):
                        self.add_paraphrases(ppword, baseword, score)
                    elif (line.split("|||")[-1].strip() == "OtherRelated"):
                        self.add_paraphrases(ppword, baseword, score)
                        self.add_paraphrases(baseword, ppword, score)
                    elif (line.split("|||")[-1].strip() == "Exclusion"):
                        self.add_paraphrases(ppword, baseword, score)
                        self.add_paraphrases(baseword, ppword, score)
                n += 1
                if n%10000 == 0:
                    print str(n) + " lines processed."
        print "Finish. Totally "+str(n)+" lines processed."

    def save_ppdb(self):
        "write ppdb2efrox"
        super(PPDB2EFROX, self).save_ppdb("ppdb_2_EFROX.txt")

class PPDB2EFROI(PPDB_2):
    def __init__(self, vocab="vocab.txt", ppdb="ppdb-2.0-tldr"):
        super(PPDB2EFROI, self).__init__(vocab, ppdb)

    def read_lexicon(self):
        print "read Equivalence Forward Reverse OtherRelated Independent"
        words = self.words
        ppdb = self.ppdb
        with open(ppdb, "r") as ppdb_f:
            lines = ppdb_f.readlines()
            print "Total lines: " + str(len(lines))
            n = 0
            for line in lines:
                if (line.split("|||")[1].strip() in words) and (line.split("|||")[2].strip() in words):
                    baseword = line.split("|||")[1].strip()
                    ppword = line.split("|||")[2].strip()
                    score = line.split("|||")[3].split(" ")[1].split("=")[1]
                    if (line.split("|||")[-1].strip() == "Equivalence"):
                        self.add_paraphrases(baseword, ppword, score)
                        self.add_paraphrases(ppword, baseword, score)
                    elif (line.split("|||")[-1].strip() == "ForwardEntailment"):
                        self.add_paraphrases(baseword, ppword, score)
                    elif (line.split("|||")[-1].strip() == "ReverseEntailment"):
                        self.add_paraphrases(ppword, baseword, score)
                    elif (line.split("|||")[-1].strip() == "OtherRelated"):
                        self.add_paraphrases(ppword, baseword, score)
                        self.add_paraphrases(baseword, ppword, score)
                    elif (line.split("|||")[-1].strip() == "Independent"):
                        self.add_paraphrases(ppword, baseword, score)
                        self.add_paraphrases(baseword, ppword, score)
                n += 1
                if n%10000 == 0:
                    print str(n) + " lines processed."
        print "Finish. Totally "+str(n)+" lines processed."

    def save_ppdb(self):
        "write ppdb2efroi"
        super(PPDB2EFROI, self).save_ppdb("ppdb_2_EFROI.txt")

class PPDB2EFRI(PPDB_2):
    def __init__(self, vocab="vocab.txt", ppdb="ppdb-2.0-tldr"):
        super(PPDB2EFRI, self).__init__(vocab, ppdb)

    def read_lexicon(self):
        print "read Equivalence Forward Reverse Independent"
        words = self.words
        ppdb = self.ppdb
        with open(ppdb, "r") as ppdb_f:
            lines = ppdb_f.readlines()
            print "Total lines: " + str(len(lines))
            n = 0
            for line in lines:
                if (line.split("|||")[1].strip() in words) and (line.split("|||")[2].strip() in words):
                    baseword = line.split("|||")[1].strip()
                    ppword = line.split("|||")[2].strip()
                    score = line.split("|||")[3].split(" ")[1].split("=")[1]
                    if (line.split("|||")[-1].strip() == "Equivalence"):
                        self.add_paraphrases(baseword, ppword, score)
                        self.add_paraphrases(ppword, baseword, score)
                    elif (line.split("|||")[-1].strip() == "ForwardEntailment"):
                        self.add_paraphrases(baseword, ppword, score)
                    elif (line.split("|||")[-1].strip() == "ReverseEntailment"):
                        self.add_paraphrases(ppword, baseword, score)
                    elif (line.split("|||")[-1].strip() == "Independent"):
                        self.add_paraphrases(ppword, baseword, score)
                        self.add_paraphrases(baseword, ppword, score)
                n += 1
                if n%10000 == 0:
                    print str(n) + " lines processed."
        print "Finish. Totally "+str(n)+" lines processed."

    def save_ppdb(self):
        "write ppdb2efri"
        super(PPDB2EFRI, self).save_ppdb("ppdb_2_EFRI.txt")

class PPDB2EFRX(PPDB_2):
    def __init__(self, vocab="vocab.txt", ppdb="ppdb-2.0-tldr"):
        super(PPDB2EFRX, self).__init__(vocab, ppdb)

    def read_lexicon(self):
        print "read Equivalence Forward Reverse Exclusive"
        words = self.words
        ppdb = self.ppdb
        with open(ppdb, "r") as ppdb_f:
            lines = ppdb_f.readlines()
            print "Total lines: " + str(len(lines))
            n = 0
            for line in lines:
                if (line.split("|||")[1].strip() in words) and (line.split("|||")[2].strip() in words):
                    baseword = line.split("|||")[1].strip()
                    ppword = line.split("|||")[2].strip()
                    score = line.split("|||")[3].split(" ")[1].split("=")[1]
                    if (line.split("|||")[-1].strip() == "Equivalence"):
                        self.add_paraphrases(baseword, ppword, score)
                        self.add_paraphrases(ppword, baseword, score)
                    elif (line.split("|||")[-1].strip() == "ForwardEntailment"):
                        self.add_paraphrases(baseword, ppword, score)
                    elif (line.split("|||")[-1].strip() == "ReverseEntailment"):
                        self.add_paraphrases(ppword, baseword, score)
                    elif (line.split("|||")[-1].strip() == "Exclusion"):
                        self.add_paraphrases(ppword, baseword, score)
                        self.add_paraphrases(baseword, ppword, score)
                n += 1
                if n%10000 == 0:
                    print str(n) + " lines processed."
        print "Finish. Totally "+str(n)+" lines processed."

    def save_ppdb(self):
        "write ppdb2efrx"
        super(PPDB2EFRX, self).save_ppdb("ppdb_2_EFRX.txt")


if __name__ == "__main__":
    if len(sys.argv)>1:
        vocab = sys.argv[1]
        lexicon = sys.argv[2]
    else:
        vocab = "vocab.txt"
        lexicon = "ppdb-2.0-tldr"
    # ppdb = PPDB2E(vocab, lexicon)
    # ppdb.read_lexicon()
    # ppdb.save_ppdb()
    # ppdb = PPDB2FR(vocab, lexicon)
    # ppdb.read_lexicon()
    # ppdb.save_ppdb()
    # ppdb = PPDB2EFR(vocab, lexicon)
    # ppdb.read_lexicon()
    # ppdb.save_ppdb()
    # ppdb = PPDB2EFRO(vocab, lexicon)
    # ppdb.read_lexicon()
    # ppdb.save_ppdb()    
    # ppdb = PPDB2EFROX(vocab, lexicon)
    # ppdb.read_lexicon()
    # ppdb.save_ppdb()    
    # ppdb = PPDB2EFROI(vocab, lexicon)
    # ppdb.read_lexicon()
    # ppdb.save_ppdb()
    ppdb = PPDB2EFRI(vocab, lexicon)
    ppdb.read_lexicon()
    ppdb.save_ppdb()  
    ppdb = PPDB2EFRX(vocab, lexicon)
    ppdb.read_lexicon()
    ppdb.save_ppdb()  
