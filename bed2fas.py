import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import NucleotideAlphabet
from Bio.Seq import Seq
import numpy as np
import csv
import argparse

parser = argparse.ArgumentParser(description='Generating Fasta File from Bed')
parser.add_argument('-p','--path', dest='path', type=str, default="/home/amato/Scrivania/CORENup/Datasets/Setting2/Yeast/Whole_genome/bed",
                    help='Bed Files Path')
parser.add_argument('-n','--nuc', dest='nucBed', type=str, default="Yeast_wg_nuc.bed",
                    help='Bed filename for Nucleosome Sequences')
parser.add_argument('-l','--lin', dest='linBed', type=str, default="Yeast_wg_link.bed",
                    help='Bed filename for Linker Sequences')
parser.add_argument('-fd','--fdir', dest='fasDir', type=str, default="/home/amato/Scrivania/CORENup/Datasets/Setting2/Yeast/fasta",
                    help='Output file Path')
parser.add_argument('-f','--fas', dest='fasName', type=str, default="nucleosomes_vs_linkers_yeast_wg.fas",
                    help='Output filename')

args = parser.parse_args()
inPath = args.path
fasPath = args.fasDir
nucBed = args.nucBed
linBed = args.linBed
fasName = args.fasName

seqList = []
print("Creating or Opening Fasta Output file")
try:
    outFp = open(os.path.join(fasPath, fasName), "w")
except:
    print("Error: Opening {} Failed".format(os.path.join(fasPath, fasName)))
    

nucName = {}
linkName = {}
print("Opening {} File".format(os.path.join(inPath, nucBed)))
try:
    csv_file = open(os.path.join(inPath, nucBed), "r")
except:
    print("Error: Opening {} Failed".format(os.path.join(inPath, nucBed)))
finally:
    print("Reading {} File".format(os.path.join(inPath, nucBed)))
    csv_reader = csv.reader(csv_file, delimiter='\t')
    line_count = 1
    for row in csv_reader:
        if(row[0] not in nucName.keys()):
            nucName[row[0]] = list(SeqIO.parse(open(os.path.join(fasPath,row[0])+".fa"),'fasta'))[0]


print("Opening {} File".format(os.path.join(inPath, linBed)))
try:
    with open(os.path.join(os.path.join(inPath, linBed)), "r") as csv_file:
        print("Reading {} File".format(os.path.join(inPath, linBed)))
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 1
        for row in csv_reader:
            if(row[0] not in linkName.keys()):
                linkName[row[0]] = list(SeqIO.parse(open(os.path.join(fasPath,row[0]+".fa")),'fasta'))[0]
except:
    print("Error: Opening {} Failed".format(os.path.join(inPath, linBed)))

perc = 0
numlines = 0
with open(os.path.join(inPath, nucBed), "r") as csv_file:
    numlines = len(csv_file.readlines())

with open(os.path.join(inPath, nucBed), "r") as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')
    line_count = 1
    for row in csv_reader:
        name = row[0]
        start = int(row[1])
        end = int(row[2])
        midpoint = int(np.ceil((end+start)/2.0))
        nuc = nucName[row[0]][(midpoint-73):(midpoint+74)]
        nuc = str(nuc.seq)
        record = SeqRecord(Seq(nuc,
                    NucleotideAlphabet),
                id="nucleosomal_sequence_"+str(line_count), name="",
                description="")
        seqList.append(record)
        line_count = line_count+1

        perc = line_count/numlines*100

        if(not line_count%10):
            print("Analizzando Nucleosomi..."+str(perc))


numlines = 0
with open(os.path.join(inPath, linBed), "r") as csv_file:
    numlines = len(csv_file.readlines())
with open(os.path.join(inPath, linBed), "r") as csv_file:

    csv_reader = csv.reader(csv_file, delimiter='\t')
    line_count = 1
    for row in csv_reader:
        name = row[0]

        start = int(row[1])
        end = int(row[2])
        midpoint = int(np.ceil((end+start)/2.0))
        link = linkName[row[0]][(midpoint-73):(midpoint+74)]
        link = str(link.seq)
        record = SeqRecord(Seq(link,
                    NucleotideAlphabet),
                id="linker_sequence_"+str(line_count), name="",
                description="")
        seqList.append(record)
        line_count = line_count+1

        perc = (line_count/numlines)*100

        if(not line_count%10):
            print("Analizzando Linker..."+str(perc))

SeqIO.write(seqList, outFp, "fasta")
