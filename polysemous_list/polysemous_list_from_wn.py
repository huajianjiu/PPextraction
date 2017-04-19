from nltk.corpus import wordnet as wn

vocbalury=[]
polysemy_list=[]

for synset in wn.all_synsets():
    w = synset.name().split(".")[0]
    if w in vocbalury:
        if w in polysemy_list:
            pass
        else:
            polysemy_list.append(w)
            # if len(polysemy_list)>5:
            #     break
    else:
        vocbalury.append(w)
for w in polysemy_list:
    print w
