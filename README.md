# Deep GONet

From the article entitled **Deep GONet: Self-explainable deep neural network based on Gene Ontology for phenotype prediction from gene expression data** (submitted to APBC 2021) by Victoria Bourgeais, Farida Zehraoui, Mohamed Ben Hamdoune, and Blaise Hanczar.

---

## Description

Deep GONet is a self-explainable neural network integrating the Gene Ontology into its hierarchical architecture.

## Get started

The code is implemented in Python using the [Tensorflow](https://www.tensorflow.org/) framework v1.12 (see [requirements.txt](https://forge.ibisc.univ-evry.fr/vbourgeais/DeepGONet/blob/master/requirements.txt) for more details)

### Dataset

The full dataset can be downloaded on ArrayExpress database under the id [E-MTAB-3732](https://www.ebi.ac.uk/arrayexpress/experiments/E-MTAB-3732/). Here, you can find the pre-processed training and test sets:

[training set](https://entrepot.ibisc.univ-evry.fr/f/5b57ab5a69de4f6ab26b/?dl=1)

[test set](https://entrepot.ibisc.univ-evry.fr/f/057f1ffa0e6c4aab9bee/?dl=1) 

Additional files for NN architecture: [filesforNNarch](https://entrepot.ibisc.univ-evry.fr/f/6f1c513798df41999b5d/?dl=1) 

### Usage

Deep GONet was achieved with the $L_{GO}$ regularization and the hyperparameter $\alpha=1e^{-2}$.  
To replicate it, the command line flag *type_training* needs to be set to LGO (default value) and the command line flag *alpha* to $1e^{-2}$ (default value).  

There exists 3 functions (flag *processing*): one is dedicated to the training of the model (*train*), another one to the evaluation of the model on the test set (*evaluate*), and the last one to the prediction of the outcomes of the samples from the test set (*predict*).

#### 1) Train


```bash
python DeepGONet.py --type_training="LGO" --alpha=1e-2 --EPOCHS=600 --is_training=True --display_step=10 --save=True --processing="train"
```

#### 2) Evaluate


```bash
python DeepGONet.py --type_training="LGO" --alpha=1e-2 --EPOCHS=600 --is_training=False --restore=True --processing="evaluate"
```

#### 3) Predict


```bash
python DeepGONet.py --type_training="LGO" --alpha=1e-2 --EPOCHS=600 --is_training=False --restore=True --processing="predict"
```

The outcomes are saved into a numpy array.

#### Help

All the details about the command line flags can be provided by the following command:


```bash
python DeepGONet.py --help
```

For most of the flags, the default values can be employed. *log_dir* and *save_dir* can be modified to your own repositories. Only the flags in the command lines displayed have to be adjusted to achieve the desired objective.

### Comparison with classical fully-connected network using L2 or L1 regularization terms

It is possible to compare the model with L2,L1 regularization instead of LGO.


```bash
python DeepGONet.py --type_training="L2" --alpha=1e-2 --EPOCHS=600 --is_training=True --display_step=10 --save=True --processing="train"
```


```bash
python DeepGONet.py --type_training="L1" --alpha=1e-2 --EPOCHS=600 --is_training=True --display_step=10 --save=True --processing="train"
```

Without regularization:


```bash
python DeepGONet.py --alpha=0 --EPOCHS=100 --is_training=True --display_step=5 --save=True --processing="train"
```

###  Interpretation tool

Please see the notebook entitled *Interpretation_tool.ipynb* to perform the biological interpretation of the results.