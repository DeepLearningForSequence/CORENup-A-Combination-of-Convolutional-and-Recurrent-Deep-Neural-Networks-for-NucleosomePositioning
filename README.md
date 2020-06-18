# CORENup
A Combination of Convolutional and Recurrent Deep Neural Networks for Nucleosome Positioning Identification

## Requirements

* python 3.7 or above
* numpy
* scikit-learn
* tensorflow 2.x
* matplotlib
* biopython
* argparse

## Data Manipulation

In order to use nucleosome datasets with CORENup neural network, you need to perform data transformations in a hot encoded compressed format using the following scripts

### Bed to Fasta file

This Script transform bed files, with nucleosome and linker position coordinates, into a fasta file with nucleosome and linker 147bp sequences.

```console

python3 bed2fas.py [-h] [-p PATH] [-n NUCBED] [-l LINBED] [-fd FASDIR]
                  [-f FASNAME]

Arguments:

  -h, --help            show this help message and exit  
  -p PATH, --path PATH  Bed Files Path  
  -n NUCBED, --nuc NUCBED  
                        Bed filename for Nucleosome Sequences  
  -l LINBED, --lin LINBED  
                        Bed filename for Linker Sequences  
  -fd FASDIR, --fdir FASDIR  
                        Output fasta file Path  
  -f FASNAME, --fas FASNAME  
                        Output fasta filename  

```

### Fasta file to Hot encoded Pickle

This Script transform the fasta file, with nucleosome and linker sequences, into pickle compressed file with hot encoded sequences

```console

python3 fas2pickle_encoded.py [-h] [-p PATH] [-f FASNAME] [-o OUTDIR]
                             [-n NUCPICKLE] [-l LINKPICKLE]

Generating Pickle encoded File from Fasta

optional arguments:
  -h, --help            show this help message and exit
  -p PATH, --path PATH  Fasta File Path
  -f FASNAME, --fas FASNAME
                        Fasta filename
  -o OUTDIR, --out OUTDIR
                        Output file Path
  -n NUCPICKLE, --nuc NUCPICKLE
                        Output Nucleosome filename
  -l LINKPICKLE, --lin LINKPICKLE
                        Output Linker filename


```

