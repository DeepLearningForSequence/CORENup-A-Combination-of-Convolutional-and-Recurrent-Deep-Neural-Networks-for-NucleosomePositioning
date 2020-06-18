import os 
from Bio import SeqIO
import pickle
import numpy as np
import argparse

################################################################
#                  Hot encoding Function                       #
################################################################
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


parser = argparse.ArgumentParser(description='Generating Pickle encoded File from Fasta')
parser.add_argument('-p','--path', dest='path', type=str, default="/home/amato/Scrivania/CORENup/Datasets/Setting2/Yeast/fasta",
                    help='Fasta File Path')
parser.add_argument('-f','--fas', dest='fasName', type=str, default="nucleosomes_vs_linkers_yeast_wg.fas",
                    help='Fasta filename')
parser.add_argument('-o','--out', dest='outDir', type=str, default="/home/amato/Scrivania/CORENup/Datasets/Setting2/Yeast/pickle",
                    help='Output file Path')
parser.add_argument('-n','--nuc', dest='nucPickle', type=str, default="nuc_Yeast_wg.pickle",
                    help='Output Nucleosome filename')
parser.add_argument('-l','--lin', dest='linkPickle', type=str, default="link_Yeast_wg.pickle",
                    help='Output Linker filename')

args = parser.parse_args()
inPath = args.path
outPath = args.outDir
fasName = args.fasName
nucPickle = args.nucPickle
linkPickle = args.linkPickle

del args

if(not os.path.isdir(outPath)):
    os.mkdir(outPath)

fastaSequences = SeqIO.parse(open(os.path.join(inPath, fasName)),'fasta')

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
print("Nucleosomi: "+str(len(nucEncList)))
print("Linker: "+str(len(linkEncList)))

with open(os.path.join(outPath, nucPickle), "wb") as fp:
    pickle.dump(nucList,fp)
with open(os.path.join(outPath, linkPickle), "wb") as fp:
    pickle.dump(linkList,fp)
with open(os.path.join(outPath, "encoded_"+nucPickle), "wb") as fp:
    pickle.dump(nucEncList,fp)
with open(os.path.join(outPath, "encoded_"+linkPickle), "wb") as fp:
    pickle.dump(linkEncList,fp)

#print(nucList)


        