# Setup

Create Environment
```
pipenv install
```

# Execution

Activate environment
```
pipenv shell
```

Run
```
python -m wealth_transfer --countries_train "IA|PK"
```

Arguments:

```
'-d' (or '--dhs_path'): Path of the dhs_final_labels.csv
'-c' (or '--countries_train'): Countries to train the data on, pipe-separated. E.g. "IA|NP|PK
'-s' (or '--si_data_path'): Path of Satellite Images Data (default  is 'data/si/')

```
# Directories
## Training Data
Train data is expected to be in `data/si/` (can be changed using `-s` argument) and grouped in one of the following directories:

```
dhs_AL_DR
dhs_EG_HT
dhs_IA_IA
dhs_ID_MZ
dhs_NG_SZ
dhs_TD_ZW
```