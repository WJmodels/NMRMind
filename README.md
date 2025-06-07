# NMRMind
NMRMind: A Transformer-based Model Enabling the Elucidation from Multidimensional NMR to Structures![image](https://github.com/user-attachments/assets/8b16e153-0b2e-41fe-9672-516635bf7dbb)


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

