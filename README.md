# NMRMind
NMRMind: A Transformer-based Model Enabling the Elucidation from Multidimensional NMR to Structures


## Environment
```
pip install -r requirements.txt
```

## Start training

```
--do_train \
$ bash script/run_nmr.sh
```

## Start inference

```
--do_test \
$ bash script/run_nmr.sh
```
NOTE: When the model is inferring, the probability of all input conditions needs to be set to 1.0

