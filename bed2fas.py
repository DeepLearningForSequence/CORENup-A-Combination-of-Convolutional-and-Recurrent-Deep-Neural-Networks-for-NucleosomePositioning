import os
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet import NucleotideAlphabet
from Bio.Seq import Seq
import numpy as np
import csv

species = ["Human_prom","Human_5U", "Yeast_prom","Mouse_prom", "Mouse_5U", "Drosophila_prom", "Drosophila_5U", "Drosophila_chr", "Human_chr", "Yeast_wg"]

for s in species:
    if(not os.path.isdir("./Resource/"+s+"/pickle")):
        os.mkdir("./Resource/"+s+"/pickle")

    print("Analizzando "+s+"...")

    seqList = []

    outFp = open("./Resource/"+s+"/fasta/nucleosomes_vs_linkers_"+s.lower()+".fas", "w")

    #Salvataggio nome file fasta

    nucName = {}
    linkName = {}

    with open("./Resource/"+s+"/bed/"+s+"_nuc.bed", "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 1
        for row in csv_reader:
            if(row[0] not in nucName.keys()):
                nucName[row[0]] = list(SeqIO.parse(open("./Resource/"+s+"/fasta/"+row[0]+".fa"),'fasta'))[0]
    
    with open("./Resource/"+s+"/bed/"+s+"_link.bed", "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        line_count = 1
        for row in csv_reader:
            if(row[0] not in linkName.keys()):
                linkName[row[0]] = list(SeqIO.parse(open("./Resource/"+s+"/fasta/"+row[0]+".fa"),'fasta'))[0]


    perc = 0
    numlines = 0
    with open("./Resource/"+s+"/bed/"+s+"_nuc.bed", "r") as csv_file:
        numlines = len(csv_file.readlines())

    with open("./Resource/"+s+"/bed/"+s+"_nuc.bed", "r") as csv_file:
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
                print("Analizzando Nucleosomi "+s+"..."+str(perc))


    numlines = 0
    with open("./Resource/"+s+"/bed/"+s+"_link.bed", "r") as csv_file:
        numlines = len(csv_file.readlines())
    with open("./Resource/"+s+"/bed/"+s+"_link.bed", "r") as csv_file:

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
                print("Analizzando Linker "+s+"..."+str(perc))

    with open("./Resource/"+s+"/fasta/nucleosomes_vs_linkers_"+s.lower()+".fas", "w") as fp:
        SeqIO.write(seqList, fp, "fasta")
