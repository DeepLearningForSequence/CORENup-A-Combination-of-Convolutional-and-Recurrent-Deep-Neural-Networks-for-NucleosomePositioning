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

## Run CORENup

### Experiment Script

You can run experiment using 3 kind of model:
1 The CORENup Architetture
2 The Conv - LSTM part only
3 The Conv - 2Conv part only

Experiments use a k-fold cross validation method to evaluete models and save models and folds in the specified output directory


```console

python3 expRun.py [-h] [-m {1,2,3} [{1,2,3} ...]] [-p PATH] [-n NUCPICKLE]
                 [-l LINKPICKLE] [-o OUTPATH] [-e EXP] [-nf NFOLDS]
                 [-f FOLDNAME]

Arguments:
  -h, --help            show this help message and exit
  -m {1,2,3} [{1,2,3} ...], --models {1,2,3} [{1,2,3} ...]
                        NN Models name
                        1 - CORENup NN
                        2 - Conv - LSTM NN
                        3 - Conv - 2Conv NN
  -p PATH, --path PATH  Pickle file Path
  -n NUCPICKLE, --nuc NUCPICKLE
                        Nucleosome filename
  -l LINKPICKLE, --lin LINKPICKLE
                        Linker filename
  -o OUTPATH, --out OUTPATH
                        Output file Path
  -e EXP, --experiments EXP
                        Experiments Name
  -nf NFOLDS, -nfolds NFOLDS
                        Number of Folds
  -f FOLDNAME, -foldName FOLDNAME
                        Folds Filename


```

### Results Plot

This script evaluate trained model using the following metrics:
* Accuracy
* Sensitivity
* Specificity
* MCC
* AUC

The metrics results bar plot was saved in the model file path with plot title name and png, svg and eps extension

```console

python3 plotExpResults.py [-h] [-m {1,2,3} [{1,2,3} ...]] [-pn PLOTNAME]
                         [-p PATH] [-e EXP] [-nf NFOLDS] [-f FOLDNAME]

Arguments:
  -h, --help            show this help message and exit
  -m {1,2,3} [{1,2,3} ...], --models {1,2,3} [{1,2,3} ...]
                        NN Models name
                        1 - CORENup NN
                        2 - Conv - LSTM NN
                        3 - Conv - 2Conv NN
  -pn PLOTNAME, --plot PLOTNAME
                        Plot Title
  -p PATH, --path PATH  Model file Path
  -e EXP, --experiments EXP
                        Experiments Name
  -nf NFOLDS, -nfolds NFOLDS
                        Number of Folds
  -f FOLDNAME, -foldName FOLDNAME
                        Folds Filename

```