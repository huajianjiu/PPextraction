import numpy as np
import os.path
from nltk.corpus import wordnet as wn
import cPickle as pickle


def remove_duplicate(list1):
    list2 = []
    for i in list1:
        if i not in list2:
            list2.append(i)
    return list2


def remove_not_in_and_get_word_id(list1, self_word, legal_word_list):
    list2 = []
    for i in list1:
        if i != self_word and i in legal_word_list and i not in list2:
            list2.append(legal_word_list.index(i))
    return list2


class WordnetNet(object):
    def __init__(self, relations_num, vocab="vocab.txt"):
        self.relations_num = relations_num
        self.vocab = vocab
        if os.path.isfile("wn_"+self.vocab+str(self.relations_num)+".pkl"):
            with open("wn_" + self.vocab + str(self.relations_num) + ".pkl", "rb") as f_load:
                self.words = pickle.load(f_load)
                self.synsets_relations = pickle.load(f_load)
                # self.scores_matrix = pickle.load(f_load)
        else:
            with open(vocab, "r") as f_vocab:
                words = f_vocab.readlines()
                words = [x.split()[0] for x in words]
                self.words = words
            self.synsets_relations = np.zeros((len(self.words), relations_num), dtype=np.int32)
            for i, word in enumerate(words):
                current_word_relations = []
                synsets = wn.synsets(word)
                for synset in synsets:
                    current_word_relations += [str(lemma.name()) for lemma in synset.lemmas()]
                    hypos = synset.hyponyms()
                    holos = synset.member_holonyms()
                    for holo in holos:
                        current_word_relations += [str(lemma.name()) for lemma in holo.lemmas()]
                current_word_relations = remove_duplicate(current_word_relations)
                current_word_relations = remove_not_in_and_get_word_id(current_word_relations, word, self.words)
                for j in range(len(current_word_relations)):
                    if j < relations_num:
                        self.synsets_relations[i][j] = current_word_relations[j]
            self.save_wordnet()

    def save_wordnet(self):
        with open("wn_"+self.vocab+str(self.relations_num)+".pkl", "wb") as f_save:
            pickle.dump(self.words, f_save)
            pickle.dump(self.synsets_relations, f_save)
            # pickle.dump(self.scores_matrix, f_save)


if __name__ == "__main__":
    wordnet_net = WordnetNet(5)
    for i in range(100):
        print str(wordnet_net.words[i+100])+":"+str(wordnet_net.synsets_relations[i+100])
