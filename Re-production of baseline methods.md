# Re-production of baseline methods

This doc describes the detail of re-production of baseline methods on both datasets. All methods were evaluated with 5-fold nested cross-validation (NCV).

## DeepSynergy

The original [paper](https://academic.oup.com/bioinformatics/article/34/9/1538/4747884?login=false), [codes](https://github.com/KristinaPreuer/DeepSynergy) and data could be found at DeepSynergy's [homepage](https://www.bioinf.jku.at/software/DeepSynergy/).

We re-implemented DeepSynergy using PyTorch according to the original paper and codes.

### O'Neil Dataset

**Drug & Cell line Features**

DeepSynergy provides the generated features at its homepage. The feature of each drug and cell line could be splitted from the given file.

**Hyperparameters**

The hyperparameters that need to be selected in NCV are as follows:

| Scenario       | Hyperparamters | Range                           | Remark                                                       |
| -------------- | -------------- | ------------------------------- | ------------------------------------------------------------ |
| LDCO, LCO, LDO | preprocessing  | norm+tanh+norm                  | The best preprocessing method according to DeepSynergy's supplementary file. |
| LDCO, LCO, LDO | # hidden unit  | 8192_8192; 8192_4096; 8192_2048 | The best threes according to DeepSynergy's supplementary file. |
| LDCO, LCO, LDO | learning rate  | 1e-4, 1e-5                      | The best twos according to DeepSynergy's supplementary file. |
| LDCO, LCO, LDO | dropout rate   | input: 0.2 + hidden: 0.5        | The best dropout rate according to DeepSynergy's supplementary file. |
| LDCO, LCO, LDO | # epoch        | 500                             | Training epochs.                                             |
| LDCO, LCO, LDO | patience       | 25                              | Early-stopping patience.                                     |

### NCI-ALMANAC Dataset

DeepSynergy didn't provide features of drugs/cell lines in NCI-ALMANAC Dataset. We generated the features follow the original paper as much as possible.

**Drug Features**

1. Remove salts of drugs using RDKit based on given drug SMILES strings. DeepSynergy did this step using OpenBabel, which we failed to installed.
2. Save processed moleculars in SDF file and get ECFP-6 drug fingerprints.
3. Get drug discriptors using [online service](http://www.scbdd.com/chemopy_desc/index/) of ChemoPy.
4. DeepSynergy omitted the detail generation step of binary toxicophore features, so we had to skip this.
5. After removing zero-variance features, we got drug features consists of 2250 ECFP_6 and 488 physico- chemical.

**Cell Line Features**

1. Microarray data were downloaded from [ArrayExpress(accession number: E-MTAB-3610)](https://www.ebi.ac.uk/biostudies/arrayexpress/studies/E-MTAB-3610?query=E-MTAB-3610).
2. Microarray data were processed using FARMS.
3. After removing zero-variance features, we got cell line features consists of 4044 gene expression.

**Hyperparameters**

The hyperparameters that need to be selected in NCV are as follows:

| Scenario       | Hyperparamters | Range                           | Remark                                                       |
| -------------- | -------------- | ------------------------------- | ------------------------------------------------------------ |
| LDCO, LCO, LDO | preprocessing  | norm+tanh+norm                  | The best preprocessing method according to DeepSynergy's supplementary file. |
| LDCO, LCO, LDO | # hidden unit  | 8192_8192; 8192_4096; 8192_2048 | The best threes according to DeepSynergy's supplementary file. |
| LDCO, LCO, LDO | learning rate  | 1e-4, 1e-5                      | The best twos according to DeepSynergy's supplementary file. |
| LDCO, LCO, LDO | dropout rate   | input: 0.2 + hidden: 0.5        | The best dropout rate according to DeepSynergy's supplementary file. |
| LDCO, LCO, LDO | # epoch        | 50                              | Training epochs reduced to 1/10 of it on the O'Neil dataset according to the number of samples. |
| LDCO, LCO, LDO | patience       | 2                               | Early-stopping patience.                                     |

## PRODeepSyn

- [paper](https://academic.oup.com/bib/article/23/2/bbab587/6511206?login=false) 
- codes and data could be found at the [repo](https://github.com/TOJSSE-iData/PRODeepSyn) 

### O'Neil Dataset

The original repo has provided required data for experiments.

**Hyperparameters**

The hyperparameters that need to be selected in NCV are as follows:

| Scenario       | Hyperparamters | Range      | Remark                                                       |
| -------------- | -------------- | ---------- | ------------------------------------------------------------ |
| LDCO, LCO, LDO | Learning rate  | 1e-3, 1e-4 | These two values are in the subset of the original paper's candidate set, where the remaining value 1e-5 always yield terrible results. |
| LDCO, LCO, LDO | Others         |            | Keep the same with the original paper.                       |

### NCI-ALMANAC Dataset

**Drug Features**

We followed the process steps in the original paper and got drug features consists of 256 Morgan_2 fingerprints and 182 descriptors.

**Cell Line Features**

1. Gene expression (GE) features were obtained as follows:
   1. Following the steps described in Section "NCI-ALMANAC Dataset,  DeepSynergy" (See above for details), we got the processed GE data of each cell line.
   2. We trained the StateEncoder followed the paper of PRODeepSyn to get cell line embedding (GE) of dim 384.
2. Mutation (MUT) features were obtained as follows:
   1. We downloaded mutation data from [COSMIC-CLP](https://cancer.sanger.ac.uk/cell_lines).
   2. We trained the StateEncoder followed the paper of PRODeepSyn to get cell line embedding (MUT) of dim 384.
3. Finally, the cell line embedding GE and MUT were concatenated and z-score normalized as cell line features.

**Hyperparameters**

The hyperparameters that need to be selected in NCV are as follows:

| Scenario       | Hyperparamters | Range      | Remark                                                       |
| -------------- | -------------- | ---------- | ------------------------------------------------------------ |
| LDCO, LCO, LDO | Learning rate  | 1e-3, 1e-4 | These two values are in the subset of the original paper's candidate set, where the remaining value 1e-5 always yield terrible results. |
| LDCO, LCO, LDO | # epoch        | 50         | Training epochs reduced to 1/10 of it on the O'Neil dataset according to the number of samples. |
| LDCO, LCO, LDO | patience       | 5          | Early-stopping patience.                                     |
| LDCO, LCO, LDO | Others         |            | Keep the same with the original paper.                       |

## TranSynergy

- [paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008653)
- codes and data could be found at the [repo](https://github.com/qiaoliuhub/drug_combination) 

### O'Neil Dataset

**Drug & Cell line Features**

TranSynergy provides the generated features at its repo. The feature of each drug and cell line could be splitted from the given file. Only **36 drugs** and **31 cell lines** were kept in experiments.

**Hyperparameters**

The hyperparameters that need to be selected in NCV are as follows:

| Scenario       | Hyperparamters     | Range                        | Remark                                  |
| -------------- | ------------------ | ---------------------------- | --------------------------------------- |
| LDCO, LCO, LDO | Cell line features | gene_expression + dependency | Best combination in the original paper. |
| LDCO, LCO, LDO | Others             |                              | Keep the same with the original paper.  |

### NCI-ALMANAC Dataset

**Cell Line Features**

1. All 43 cell lines' gene expression data were available at [Harmonizome](https://maayanlab.cloud/Harmonizome/dataset/CCLE+Cell+Line+Gene+Expression+Profiles) 
2. 33 of 43 cell lines' dependency data were availabel at [DepMap](https://depmap.org) (23Q2)
3. Use MICE to complete missing cell line features as TranSynergy did.

**Drug Features**

1. We collected drug-target interaction from [DrugBank](https://go.drugbank.com/releases/latest) and [ChEMBL](https://www.ebi.ac.uk/chembl/). As results, we kept **87** of 102 drugs that have target.
2. We remained 2160 of 2197 cancer genes (provided by TranSynergy) whose expression data or dependency data was available.
3. 197 of 202 drug targets could be mapped to PPI.
4. Use [network_propagation.py](https://github.com/idekerlab/pyNBS/blob/master/pyNBS/network_propagation.py) in pyNBS to generate drug features according to TranSynergy.

**Hyperparameters**

The hyperparameters that need to be selected in NCV are as follows:

| Scenario       | Hyperparamters     | Range                        | Remark                                  |
| -------------- | ------------------ | ---------------------------- | --------------------------------------- |
| LDCO, LCO, LDO | Cell line features | gene_expression + dependency | Best combination in the original paper. |
| LDCO, LCO, LDO | # epoch            | 80                           | 1/10 of it in O'Neil                    |
| LDCO, LCO, LDO | Others             |                              | Keep the same with the original paper.  |

## MGAE-DC

- [paper](https://doi.org/10.1371/journal.pcbi.1010951)
- codes and data could be found at the [repo](https://github.com/yushenshashen/MGAE-DC)

### O'Neil Dataset

**Drug Features**

1. Drug fingerprints were provided by MGAE-DC
2. Since the number of cell lines is different from the original paper (34 vs 39), the cell line-specific and cell line-common features need to be trained from scratch.

**Cell Line Features**

1. Cell line features could be found at the repo.

**Hyperparameters**

The hyperparameters that need to be selected in NCV are as follows:

| Scenario       | Hyperparamters                    | Range        | Remark                                                      |
| -------------- | --------------------------------- | ------------ | ----------------------------------------------------------- |
| LDCO, LCO, LDO | Dimensionality   of the embedding | 320          | Best one given in the original paper.                       |
| LDCO, LCO, LDO | Hidden size                       | 8192         | Best one given in the original paper and code.              |
| LDCO, LCO, LDO | learning rate                     | 0.001 0.0001 | Best one given in the original paper; Best one in the code. |
| LDCO, LCO, LDO | dropout                           | 0.2          | Best one given in the original paper.                       |

### NCI-ALMANAC Dataset

**Drug Features**

1. Drug informax fingerprints generated using [pretrain-gnns](https://github.com/snap-stanford/pretrain-gnns/tree/master) based on SMILES string, according to MGAE-DC's code.
2. Since the number of cell lines is different from the original paper, the cell line-specific and cell line-common features need to be trained from scratch.

**Cell Line Features**

1. Cell line features could be found at the repo.

**Hyperparameters**

The hyperparameters that need to be selected in NCV are as follows:

| Scenario       | Hyperparamters                    | Range        | Remark                                                      |
| -------------- | --------------------------------- | ------------ | ----------------------------------------------------------- |
| LDCO, LCO, LDO | Dimensionality   of the embedding | 320          | Best one given in the original paper.                       |
| LDCO, LCO, LDO | Hidden size                       | 8192         | Best one given in the original paper and code.              |
| LDCO, LCO, LDO | learning rate                     | 0.001 0.0001 | Best one given in the original paper; Best one in the code. |
| LDCO, LCO, LDO | dropout                           | 0.2          | Best one given in the original paper.                       |

## HypergraphSynergy

- [paper](https://academic.oup.com/bioinformatics/article/38/20/4782/6674505)
- codes and data could be found at the [repo](https://github.com/liuxuan666/HypergraphSynergy)

### O'Neil Dataset

**Drug Features**

1. Drug fingerprints were provided by HypergraphSynergy

**Cell Line Features**

1. HypergraphSynergy provided 32 cell lines' features, which could not cover the 34 cell lines used in our dataset. We downloaded gene expression data from [COSMIC-CLP](https://cancer.sanger.ac.uk/cell_lines) and selected 651 genes' data according to HypergraphSynergy as cell line features.

**Hyperparameters**

The hyperparameters that need to be selected in NCV are as follows:

| Scenario       | Hyperparamters | Range     | Remark                                                       |
| -------------- | -------------- | --------- | ------------------------------------------------------------ |
| LDCO, LCO, LDO | # epochs       | [1, 4000] | The original paper use 4000 as the number of epochs as it yielded the best result on test set. However, the test set should be seperated from the training progress. So we selected the number of epochs according to metrics on validation set. |

Others

| **Module**                         | **Layers**              | **Output shape** | **Operation**                                                | **Description**                                              |
| :--------------------------------- | :---------------------- | :--------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Drug   representation              | Drug_input              | *N×n×*75         |                                                              | Atoms node feature   (n is the number of atoms)              |
|                                    |                         | *N×n×n*          |                                                              | Atoms adjacent  matrix                                       |
|                                    | GCN_layer               | *N×n×*128        | Kernel size =   (75,128)                                     | Drug feature extraction                                      |
|                                    |                         |                  | ReLu activation, Batch normalization,   Dropout(0.3)         |                                                              |
|                                    | GCN_layer               | *N×n×*100        | Kernel size =   (128,100)                                    |                                                              |
|                                    |                         |                  | ReLu activation function,   Batch normalization              |                                                              |
|                                    | Global max   pooling    | *N×*100          |                                                              |                                                              |
| Cell line   representation         | Genomic_input           | *M×*651          |                                                              | Genomic feature   extraction                                 |
|                                    | Fully connected   layer | *M×*128          | Tanh activation function, Batch   normalization, Dropout(0.3) |                                                              |
|                                    | Fully connected   layer | *M×*100          | Relu activation   function                                   |                                                              |
| Hypergraph   neural network (HGNN) | Cell line   node_input  | *M×*100          |                                                              | Output of Cell line   representation                         |
|                                    | Drug node_input         | *N×*100          |                                                              | Output of Drug   representation                              |
|                                    | Incident matrix input   | 2*×E*            |                                                              | *E* is the edge number                                       |
|                                    | Weight matrix           | *E×E*            |                                                              | The hyperedge weight matrix                                  |
|                                    | HGNN_layer              | (*M+N*)×256      | Kernel size =   (100,256)                                    | HGNN encoder                                                 |
|                                    |                         |                  | LeakyReLU   activation function                              |                                                              |
|                                    |                         |                  | Batch   normalization                                        |                                                              |
|                                    | HGNN_layer              | (*M+N*)×256      | Kernel size =   (256,256)                                    |                                                              |
|                                    |                         |                  | LeakyReLU   activation function                              |                                                              |
|                                    |                         |                  | Batch   normalization                                        |                                                              |
|                                    | HGNN_layer              | (*M+N*)×256      | Kernel size = (256,256) ReLU activation function             |                                                              |
|                                    |                         |                  | None                                                         |                                                              |
| Scoring function                   | Node_input              | (*M+N*)×256      |                                                              | Output of HGNN                                               |
|                                    | Concatenation           | *B×*768          |                                                              | Concatenation   drug-drug-cell line triplets representation, *B* is the batch size number |
|                                    | Fully connected   layer | *B×*384          | LeakyReLU activation function, Batch normalization           |                                                              |
|                                    |                         |                  | Dropout (0.4)                                                |                                                              |
|                                    | Fully connected   layer | *B×*192          | Tanh activation function (LeakyReLU   activation function), Batch normalization |                                                              |
|                                    |                         |                  | Drop out(0.4)                                                |                                                              |
|                                    | Fully connected   layer | *B×*1            | None                                                         | Predict the   probability/score of their synergistic effect  |

> Note: *M* is the number of Cell line nodes, *N* is the number of Drug nodes, *E* is the number of edges and *B* is the batch size (set as 64). The numbers of GCN layers and FCN layers were both set to 3. The number of HGNN layers was fixed to 2, and the hyperedge weight matrix *W* is set to the identity matrix of the appropriate dimensions because we assume that the contribution of each hyperedge in the hypergraph is the same.

### NCI-ALMANAC Dataset

**Drug Features**

1. We supplemented SMILES strings of drugs and got drug features according to HypergraphSynergy.

**Cell Line Features**

1. HypergraphSynergy provided 55 cell lines' features, which could not cover the 43 cell lines used in our dataset. We downloaded gene expression data from [COSMIC-CLP](https://cancer.sanger.ac.uk/cell_lines) and selected 651 genes' data according to HypergraphSynergy as cell line features.

**Hyperparameters**

The hyperparameters that need to be selected in NCV are as follows:

| Scenario       | Hyperparamters                    | Range        | Remark                                                      |
| -------------- | --------------------------------- | ------------ | ----------------------------------------------------------- |
| LDCO, LCO, LDO | Dimensionality   of the embedding | 320          | Best one given in the original paper.                       |
| LDCO, LCO, LDO | Hidden size                       | 8192         | Best one given in the original paper and code.              |
| LDCO, LCO, LDO | learning rate                     | 0.001 0.0001 | Best one given in the original paper; Best one in the code. |
| LDCO, LCO, LDO | dropout                           | 0.2          | Best one given in the original paper.                       |

**Hyperparameters**

The hyperparameters that need to be selected in NCV are as follows:

| Scenario       | Hyperparamters | Range     | Remark                                                       |
| -------------- | -------------- | --------- | ------------------------------------------------------------ |
| LDCO, LCO, LDO | # epochs       | [1, 4000] | The original paper use 4000 as the number of epochs as it yielded the best result on test set. However, the test set should be seperated from the training progress. So we selected the number of epochs according to metrics on validation set. |

Others

| **Module**                         | **Layers**              | **Output shape** | **Operation**                                                | **Description**                                              |
| :--------------------------------- | :---------------------- | :--------------- | :----------------------------------------------------------- | :----------------------------------------------------------- |
| Drug   representation              | Drug_input              | *N×n×*75         |                                                              | Atoms node feature   (n is the number of atoms)              |
|                                    |                         | *N×n×n*          |                                                              | Atoms adjacent  matrix                                       |
|                                    | GCN_layer               | *N×n×*128        | Kernel size =   (75,128)                                     | Drug feature extraction                                      |
|                                    |                         |                  | ReLu activation, Batch normalization,   Dropout(0.3)         |                                                              |
|                                    | GCN_layer               | *N×n×*100        | Kernel size =   (128,100)                                    |                                                              |
|                                    |                         |                  | ReLu activation function,   Batch normalization              |                                                              |
|                                    | Global max   pooling    | *N×*100          |                                                              |                                                              |
| Cell line   representation         | Genomic_input           | *M×*651          |                                                              | Genomic feature   extraction                                 |
|                                    | Fully connected   layer | *M×*128          | Tanh activation function, Batch   normalization, Dropout(0.3) |                                                              |
|                                    | Fully connected   layer | *M×*100          | Relu activation   function                                   |                                                              |
| Hypergraph   neural network (HGNN) | Cell line   node_input  | *M×*100          |                                                              | Output of Cell line   representation                         |
|                                    | Drug node_input         | *N×*100          |                                                              | Output of Drug   representation                              |
|                                    | Incident matrix input   | 2*×E*            |                                                              | *E* is the edge number                                       |
|                                    | Weight matrix           | *E×E*            |                                                              | The hyperedge weight matrix                                  |
|                                    | HGNN_layer              | (*M+N*)×256      | Kernel size =   (100,256)                                    | HGNN encoder                                                 |
|                                    |                         |                  | LeakyReLU   activation function                              |                                                              |
|                                    |                         |                  | Batch   normalization                                        |                                                              |
|                                    | HGNN_layer              | (*M+N*)×256      | Kernel size =   (256,256)                                    |                                                              |
|                                    |                         |                  | LeakyReLU   activation function                              |                                                              |
|                                    |                         |                  | Batch   normalization                                        |                                                              |
|                                    | HGNN_layer              | (*M+N*)×256      | Kernel size = (256,256) ReLU activation function             |                                                              |
|                                    |                         |                  | None                                                         |                                                              |
| Scoring function                   | Node_input              | (*M+N*)×256      |                                                              | Output of HGNN                                               |
|                                    | Concatenation           | *B×*768          |                                                              | Concatenation   drug-drug-cell line triplets representation, *B* is the batch size number |
|                                    | Fully connected   layer | *B×*384          | LeakyReLU activation function, Batch normalization           |                                                              |
|                                    |                         |                  | Dropout (0.4)                                                |                                                              |
|                                    | Fully connected   layer | *B×*192          | Tanh activation function (LeakyReLU   activation function), Batch normalization |                                                              |
|                                    |                         |                  | Drop out(0.4)                                                |                                                              |
|                                    | Fully connected   layer | *B×*1            | None                                                         | Predict the   probability/score of their synergistic effect  |

> Note: *M* is the number of Cell line nodes, *N* is the number of Drug nodes, *E* is the number of edges and *B* is the batch size (set as 64). The numbers of GCN layers and FCN layers were both set to 3. The number of HGNN layers was fixed to 2, and the hyperedge weight matrix *W* is set to the identity matrix of the appropriate dimensions because we assume that the contribution of each hyperedge in the hypergraph is the same.

## DeepDDS

- [paper](https://academic.oup.com/bib/article/23/1/bbab390/6375262)
- codes and data could be found at the [repo](https://github.com/Sinwang404/DeepDDS/tree/master)

### O'Neil Dataset

**Drug Features**

1. We supplemented SMILES from DrugBank and generated features with codes in the repo.

**Cell Line Features**

1. Cell line features could be generated with codes in the repo.

**Hyperparameters**

The hyperparameters that need to be selected in NCV are as follows:

| Scenario       | Hyperparamters | Range        | Remark            |
| -------------- | -------------- | ------------ | ----------------- |
| LDCO, LCO, LDO | # fc dim       | 1024 512 128 | Given by the code |
| LDCO, LCO, LDO | # mlp dim      | 1024 512     | Given by the code |
| LDCO, LCO, LDO | learning rate  | 0.001        | Given by the code |
| LDCO, LCO, LDO | # epochs       | 1000         | Given by the code |

### NCI-ALMANAC Dataset

**Drug Features**

1. We supplemented SMILES from DrugBank and generated features with codes in the repo.

**Cell Line Features**

1. Cell line features could be generated with codes in the repo.

**Hyperparameters**

The hyperparameters that need to be selected in NCV are as follows:

| Scenario       | Hyperparamters | Range        | Remark                 |
| -------------- | -------------- | ------------ | ---------------------- |
| LDCO, LCO, LDO | # fc dim       | 1024 512 128 | Given by the code      |
| LDCO, LCO, LDO | # mlp dim      | 1024 512     | Given by the code      |
| LDCO, LCO, LDO | learning rate  | 0.001        | Given by the code      |
| LDCO, LCO, LDO | # epochs       | 100          | 1/10 of O'Neil dataset |
