# MMFSynergy

This project is the source code of "Fusing Micro- and Macro-Scale Information to Predict Anticancer Synergistic Drug Combinations"

## Envs

The project requires several modules as follows:

- torch 1.10.0+
- dgl 0.7+
- PyYAML 6.0
- rdkit 2023.3.2
- transformers 4.28.1

Use `pip3 install -r requirements.txt` to install modules.


## Fast run

We provided several bash scripts to run experiments.

- run `python3 train_tokenizer.py configs/{protein_aa/drug_smiles}_tokenizer.yml` to train AA/SMILES tokenizers, see the code for detail arguments.
- run\_exp\_{aa/smiles}\_encoders\_{mlm/simcse}.sh: Train micro encoders using AA sequences or SMILES strings with MLM/SimCSE task.
- run\_exp\_macro.sh: Train the macro encoder.
- run_exp_fusion: Fuse micro and macro information to generate features.
- run_ncv.sh: Run starified nested cross-validation.

## Detailed configs

Configs used in experiments are stored under the `configs` directory.

- {protein_aa/drug_smiles}\_tokenizer.yml: Train tokenizers using AA sequences or SMILES strings.
- {protein_aa/drug_smiles}\_encoder.yml: Train micro encoder using AA sequences or SMILES strings with MLM task.
- {protein_aa/drug_smiles}\_encoder\_simcse.yml: Train micro encoder using AA sequences or SMILES strings with SimCSE task.
- macro.yml: Train macro encoder
- fuse_{protein/drug}.yml: Fuse micro and macro information of protein/drug
- nested_cv.yml: Run starified nested cross-validation.


# Licence

This work is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
