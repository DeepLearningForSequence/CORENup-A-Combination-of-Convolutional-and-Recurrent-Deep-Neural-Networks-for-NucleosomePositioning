# CORENup
A Combination of Convolutional and Recurrent Deep Neural Networks for Nucleosome Positioning Identification

## Data Manipulation

In order to use nucleosome datasets with CORENup neural network, you need to perform data transformations in a hot encoded compressed format using the following scripts

### Bed to Fasta file

This Script transform bed files, with nucleosome and linker position coordinates, into a fasta file with nucleosome and linker 147bp sequences.

```console

bed2fas.py [-h] [-p PATH] [-n NUCBED] [-l LINBED] [-fd FASDIR]
                  [-f FASNAME]

```

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