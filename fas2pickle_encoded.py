import os 
from Bio import SeqIO
import pickle
import numpy as np

def hot_encode(sequence):
    seq_encoded = np.zeros((len(sequence),4))
    dict_nuc = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T":3
    }
    i = 0
    for l in sequence:
        if(l.upper() in dict_nuc.keys()):
            seq_encoded[i][dict_nuc[l.upper()]] = 1
            i = i+1
        else:
            return []
    return seq_encoded

species=["Elegans", "Melanogaster", "Sapiens", "Yeast"]

for s in species:
    if(not os.path.isdir("./Resource/"+s+"/pickle")):
        os.mkdir("./Resource/"+s+"/pickle")

    fastaSequences = SeqIO.parse(open("./Resource/"+s+"/fasta/nucleosomes_vs_linkers_"+s.lower()+".fas"),'fasta')

    nucList = []
    linkList = []
    nucEncList = []
    linkEncList = []
    
    for fasta in fastaSequences: 
        name, sequence = fasta.id, str(fasta.seq)
        if "nucleosomal" in name:
            nucList.append(sequence)
            aus_seq = hot_encode(sequence)
            if(len(aus_seq) != 0):
                nucEncList.append(aus_seq)
        else:
            linkList.append(sequence)
            aus_seq = hot_encode(sequence)
            if(len(aus_seq) != 0):
                linkEncList.append(aus_seq)
    print("Nucleosomi "+s+":"+str(len(nucEncList)))
    print("Linker "+s+":"+str(len(linkEncList)))

    with open("./Resource/"+s+"/pickle/nucleosome_"+s+".pickle", "wb") as fp:
        pickle.dump(nucList,fp)
    with open("./Resource/"+s+"/pickle/linker_"+s+".pickle", "wb") as fp:
        pickle.dump(linkList,fp)
    with open("./Resource/"+s+"/pickle/nucleosome_encode_"+s+".pickle", "wb") as fp:
        pickle.dump(nucEncList,fp)
    with open("./Resource/"+s+"/pickle/linker_encode_"+s+".pickle", "wb") as fp:
        pickle.dump(linkEncList,fp)

#print(nucList)
    

           