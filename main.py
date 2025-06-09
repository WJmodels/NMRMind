#!/usr/bin/env python
# coding=utf-8
import os
import sys
main_path = os.path.dirname(os.path.abspath(__file__))
print("main_path", main_path)
sys.path.append(main_path)

import argparse
import logging
import math
from operator import imod
import json
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from Mydataset import MyDataset
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import time
# from tqdm.auto import tqdm
import torch.distributed as dist
import pprint
from func_timeout import func_set_timeout
import copy
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from tqdm import tqdm

import transformers
from transformers import (
    MODEL_MAPPING,
    AdamW,
    SchedulerType,
    get_scheduler,
    set_seed,
)
from transformers.utils.versions import require_version
from my_models.multi_constraint_molecular_generator import MultiConstraintMolecularGenerator
import wandb
from pytorch_lightning.lite import LightningLite
import ipdb
import torch
from torch import nn
import random
import datetime
import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

from transformers import AutoModelForCausalLM, AutoTokenizer

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    # parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument('--backend', type=str, default='nccl', choices=['gloo', 'nccl'])
    parser.add_argument("--num_nodes", type=int, default=1, help="num_nodes.") 
    parser.add_argument("--precision", type=str, default="fp32", help="precision")
    parser.add_argument("--train_folder", type=str, default="Dataset/nmr/train_data_0124_zinc.json")
    parser.add_argument("--validation_folder", type=str, default=None)
    parser.add_argument("--test_folder", type=str, default="Dataset/50k_class/testclass.json")
    parser.add_argument('--mode_for_test', type=str, default='best', choices=['best', 'last'])
    parser.add_argument("--mode", type=str, default="forward", choices=["reverse", "forward"], 
                        help="reverse, smiles-->nmr; forward: nmr-->smiles")
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--length_penalty", type=float, default=0.4, help="length_penalty")
    parser.add_argument("--use_smiles_prob", type=float, default=0.8)
    parser.add_argument("--use_molecular_formula_prob", type=float, default=1.0)
    parser.add_argument("--use_molecular_weight_prob", type=float, default=0.0)
    parser.add_argument("--use_fragment_prob", type=float, default=0.0)
    parser.add_argument("--aug_nmr", action="store_true", default=False)
    parser.add_argument("--use_13C_NMR_prob", type=float, default=1.0)
    parser.add_argument("--use_1H_NMR_prob", type=float, default=1.0)
    parser.add_argument("--use_COSY_prob", type=float, default=1.0)
    parser.add_argument("--use_HMBC_prob", type=float, default=1.0)
    parser.add_argument("--use_HH_prob", type=float, default=1.0)
    # parser.add_argument("--aug_smiles", action="store_true", default=False)
    parser.add_argument("--flag_dual", action="store_true", default=False)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--config_json_path", type=str, default="configs/bart.json")
    parser.add_argument("--tokenizer_path", type=str, default="./tokenizer/tokenizer-smiles-bart/")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. \
                            Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--model_weight", type=str, default="/home/sunhnayu/jupyterlab/XXI/syn-branch_0906/weight/5000W/epoch_1_loss_0.040273.pth", help="model_weight")
    parser.add_argument("--do_train", action="store_true", default=False)
    parser.add_argument("--do_test", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size",)
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Initial learning rate (after the potential warmup period) to use.",)
    parser.add_argument("--weight_decay", type=float, default=0.0001, help="Weight decay to use.")

    parser.add_argument("--num_train_epochs", type=int, default=200, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps to perform. If provided, overrides num_train_epochs.",)

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)

    parser.add_argument("--lr_scheduler_type", type=SchedulerType, default="linear", help="The scheduler type to use.",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"])
    parser.add_argument("--num_warmup_epochs", type=int, default=1, help="Number of epochs for the warmup in the lr scheduler.")

    parser.add_argument("--output_dir", type=str, default="weight", help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--num_workers", type=int, default=8, help="The number of processes to use for the preprocessing.")
    parser.add_argument("--input_name", type=str, default="13C_NMR")
    parser.add_argument("--output_name", type=str, default="smiles,")

    args = parser.parse_args()
    
    args.num_workers = max(args.num_workers, args.gpus)
    args.num_workers=0
    
    args.input_name = [_ for _ in args.input_name.split(",") if _ != ""]
    args.output_name = [_ for _ in args.output_name.split(",") if _ != ""]
    
    if args.precision == "fp32":
        args.fp32 = True
    else:
        args.fp32 = False
    
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    return args

# @func_set_timeout(120)
def evaluate_result(pred_smiles, origin_smiles, rdkit_conf={}):
    # Chem = rdkit_conf["Chem"]
    # AllChem = rdkit_conf["AllChem"]
    # DataStructs = rdkit_conf["DataStructs"]
    acc = 0
    valid = 0
    similarity = 0
    
    pred_mol = None
    try:
        pred_mol = Chem.MolFromSmiles(pred_smiles)
        canonical_smiles = Chem.MolToSmiles(pred_mol)
    except Exception as e:
        return acc, valid, similarity
    
    if pred_mol is None:
        return acc, valid, similarity
    else:
        valid = 1
        origin_mol = Chem.MolFromSmiles(origin_smiles)
        
        if Chem.MolToSmiles(pred_mol) == Chem.MolToSmiles(origin_mol):
            acc = 1
        
        try:
            fp_1, fp_2 = AllChem.GetMorganFingerprint(pred_mol, 2, useChirality=True), AllChem.GetMorganFingerprint(origin_mol, 2, useChirality=True)
            similarity = DataStructs.TanimotoSimilarity(fp_1, fp_2)
        except Exception as e:
            similarity = 0.0
            print(e)
            # ipdb.set_trace()
        
        return acc, valid, similarity

# @func_set_timeout(30)
def get_score(pred_smiles_list, origin_smiles_list):
    acc_count = 0
    valid_count = 0
    tanimoto_sim = 0.0
    count = 0
    for i in range(len(pred_smiles_list)):
        count = count + 1
        pred_smiles = pred_smiles_list[i]
        origin_smiles = origin_smiles_list[i]
        try:
            acc, valid, similarity = evaluate_result(pred_smiles, origin_smiles)
            acc_count += acc
            valid_count += valid
            tanimoto_sim += similarity
        except Exception as e:
            print(e)
            continue
        
    acc = acc_count * 1.0 / count
    valid = valid_count * 1.0 / count
    tanimoto_sim = tanimoto_sim *1.0 /count
    
    return {
        "accurancy": round(acc, 3),
        "valid": round(valid,3),
        "tanimoto_sim":round(tanimoto_sim,3),
        "count": count,
    }
    
def canonicalized_smiles(smiles):
    smiles = smiles.replace("</SMILES>","").replace("<SMILES>","").replace("+","")
    mol = None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception as e:
        print(e)
        return None
    
    if mol is not None:
        try:
            canonical_smiles = Chem.MolToSmiles(mol)
            return canonical_smiles
        except Exception as e:
            print(e)
            return None
    else:
        right_index = len(smiles)-1
        while right_index>=0:
            if smiles[right_index] in ["."]:
                right_index = right_index - 1
            else:
                break
        left_index = 0
        while left_index<right_index:
            if smiles[left_index] in ["."]:
                left_index = left_index + 1
            else:
                break
        
        new_smiels = smiles[left_index : right_index+1]
        mol = Chem.MolFromSmiles(new_smiels)
        if mol is not None:
            try:
                canonical_smiles = Chem.MolToSmiles(mol)
                return canonical_smiles
            except Exception as e:
                print(e)
                return None
    
    return None

def get_topk_accuracy_for_single(label, predict_list=[], topk=1):
    for i in range(len(predict_list)):
        if i == topk:
            break
        elif i < topk:
            predict = predict_list[i]
            if predict is None:
                continue
            try:
                predict_mol = Chem.MolFromSmiles(predict)
                if predict_mol is None:
                    continue
                else:
                    true_mol = Chem.MolFromSmiles(label)
                    if Chem.MolToSmiles(predict_mol) == Chem.MolToSmiles(true_mol):
                        return 1
            except:
                continue
    return 0

def get_topk_accuracy(label_list, predict_list_list=[[]], topk=1):
    assert len(label_list) == len(predict_list_list)
    total = len(label_list)
    accuracy = 0
    for i in range(len(label_list)):
        label = label_list[i]
        predict_list = predict_list_list[i]
        result = get_topk_accuracy_for_single(label, predict_list, topk)
        accuracy = accuracy + result
    return accuracy*1.0/(total), accuracy, total

def get_topk_tanimoto_for_single(label, predict_list=[], topk=1):
    
    total_sim = 0.0
    
    for i in range(len(predict_list)):
        if i == topk:
            break
        elif i < topk:
            predict = predict_list[i]
            if predict is None:
                continue
            try:
                predict_mol = Chem.MolFromSmiles(predict)
                if predict_mol is None:
                    continue
                else:
                    origin_mol = Chem.MolFromSmiles(label)
                    fp_1, fp_2 = AllChem.GetMorganFingerprint(predict_mol, 2, useChirality=True), AllChem.GetMorganFingerprint(origin_mol, 2, useChirality=True)
                    similarity = DataStructs.TanimotoSimilarity(fp_1, fp_2)
                    total_sim = total_sim + similarity
            except:
                continue
    return 0


def get_topk_tanimoto(label_list, predict_list_list=[[]], topk=1):
    assert len(label_list) == len(predict_list_list)
    total = len(label_list)
    total_sim = 0
    for i in range(len(label_list)):
        label = label_list[i]
        predict_list = predict_list_list[i]
        result = get_topk_tanimoto_for_single(label, predict_list, topk)
        if result!=result:
            result = 0
        total_sim = total_sim + result
    return total_sim*1.0/(total)/topk, total_sim, total


def get_smiles(tokenizer, outputs):
    result = outputs.logits.argmax(-1)
    pred_smiles = []
    for _ in range(len(result)):
        smiles = [tokenizer.decode(i) for i in result[_] if i<202] #if i<202
        ## add filter of <SMILES> and </SMILES> 
        smiles = [i.replace("<CLASS>", "").replace("</CLASS>", "").replace("<SMILES>", "").replace("</SMILES>", "").replace("<MATERIALS>", "").replace("</MATERIALS>", "").replace("</QED>", "").replace("<QED>", "").replace("<logP>", "").replace("</logP>", "").replace("<pad>", "").replace("</s>", "").replace("</fragment>", "").replace("<fragment>", "").replace("<SA>", "").replace("</SA>", "").replace("<mask>", "") for i in smiles]
        smiles = "".join(smiles)
        pred_smiles.append(smiles)
    return pred_smiles

def get_smiles_from_generation(args, model, batch, tokenizer, kwargs={}, local_rank=-1):
    if "labels" in batch:
        del batch["labels"]
    if "decoder_input_ids" in batch:
        del batch["decoder_input_ids"]
    batch.update(kwargs)
    batch["num_beams"] = 1
    
    if '1H_NMR' not in args.output_name and '13C_NMR' not in args.output_name:  
        if args.mode == "reverse":
            # smile——>nmr
            if local_rank >= 0: 
                result = model.module.infer_2(**batch)["smiles"] # List
            else:
                result = model.infer_2(**batch)["smiles"] # List
        else:
            # nmr——>smiles
            if local_rank >= 0: 
                result = model.module.infer(**batch)["smiles"] # List
            else:
                result = model.infer(**batch)["smiles"] # List
    else:
        # smile——>nmr
        batch["bos_token_id"] = tokenizer.convert_tokens_to_ids("<1H_NMR>")
        batch["eos_token_id"] = tokenizer.convert_tokens_to_ids("</13C_NMR>")
        if local_rank >= 0: 
            result = model.module.infer_nmr(**batch) # List
        else:
            result = model.infer_nmr(**batch) # List
    return result

def get_dist_score(gathered_score_list):
    score  = {
        "accurancy": 0,
        "valid": 0,
        "tanimoto_sim":0,
        "count":0,
    }
    for temp_score in gathered_score_list:
        score["accurancy"] += temp_score["accurancy"] * temp_score["count"]
        score["valid"] += temp_score["valid"] * temp_score["count"]
        score["tanimoto_sim"] += temp_score["tanimoto_sim"] * temp_score["count"]
        score["count"] += temp_score["count"]
    score["accurancy"] = round(score["accurancy"]/score["count"], 3)
    score["valid"] = round(score["valid"]/score["count"], 3)
    score["tanimoto_sim"] = round(score["tanimoto_sim"]/score["count"], 3)
    return score

def test(args, model, test_dataloader, kwargs, local_rank, tokenizer, device, use_best_model=False):
    """_summary_

    Args:
        args (namedspace): _description_
        model (nn.module): _description_
        test_dataloader (_type_): _description_
        kwargs (_type_): _description_
        local_rank (_type_): _description_
        tokenizer (_type_): _description_
        device (_type_): _description_
        use_best_model (bool, optional): _description_. Defaults to False.
    """




    best_path = os.path.join(args.output_dir, "best.pth")
    if use_best_model and os.path.exists(best_path):
        model_dict = torch.load(best_path, map_location=device)
        model.load_state_dict(model_dict, strict=False)
        print('load weights successfully')
    
    result_list = []
    model.eval()

    for step, (idx_list, smiles_list, batch) in tqdm(enumerate(test_dataloader)):
        for k,v in batch.items():
            batch[k] = v.to(device)
        if "labels" in batch:
            del batch["labels"]
        if "decoder_input_ids" in batch:
            decoder_input_ids = batch.pop("decoder_input_ids")
        batch.update(kwargs)

            
        with torch.cuda.amp.autocast(enabled=args.fp32):
            with torch.no_grad():
                if '1H_NMR' not in args.output_name and '13C_NMR' not in args.output_name:
                    
                    if args.mode == "reverse":
                        # smile——>nmr
                        if local_rank >= 0: 
                            result = model.module.infer_2(**batch)["smiles"] # List
                        else:
                            result = model.infer_2(**batch)["smiles"] # List

                    else:
                        # import ipdb
                        # ipdb.set_trace()
                        # nmr——>smiles
                        if local_rank >= 0:

                            result = model.module.infer(**batch)["smiles"] # List
                        else:
                            result = model.infer(**batch)["smiles"] # List
                    
                else:

                    # smile——>nmr
                    batch["bos_token_id"] = tokenizer.convert_tokens_to_ids("<1H_NMR>")
                    batch["eos_token_id"] = tokenizer.convert_tokens_to_ids("</13C_NMR>")
                    if local_rank >= 0: 
                        result = model.module.infer_nmr(**batch)['smiles'] # List
                    else:
                        result = model.infer_nmr(**batch)
                        
                torch.cuda.empty_cache()
        
        for _, idx in enumerate(idx_list):
            if args.mode == "reverse":
                result_list.append(
                    {
                        "sub_smiles": [smiles_list[_]],
                        "index": idx,
                        "result": result[(_)*args.num_beams: (_+1)*args.num_beams]
                    }
                )
            else:
                result_list.append(
                    {
                        "smiles": [smiles_list[_]],
                        "index": idx,
                        "result": result[(_)*args.num_beams: (_+1)*args.num_beams]
                    }
                )

                
    if local_rank >= 0: 
        gathered_result_list = [None for i in range(dist.get_world_size())]
        dist.all_gather_object(gathered_result_list, result_list)
        
        new_result_list = []
        for temp_result_list in gathered_result_list:
            new_result_list.extend(temp_result_list)
        
        result_list = new_result_list

    if local_rank <= 0:
        if use_best_model is True:
            file_name = f"result_best_{args.num_beams}.json"
        else:
            print('meidaoru_best')
            file_name = f"result_{args.num_beams}.json"
        with open(os.path.join(args.output_dir, file_name), "w") as f:
            json.dump(result_list, f, indent=4)
        
        label_list = []
        predict_list_list = []
        for result_dict in result_list:
            if args.mode == "reverse":
                if isinstance(result_dict["sub_smiles"], str):
                    ref_smiles = result_dict["sub_smiles"]
                elif isinstance(result_dict["sub_smiles"], list):
                    ref_smiles = result_dict["sub_smiles"][0]
            else:
                if isinstance(result_dict["smiles"], str):
                    ref_smiles = result_dict["smiles"]
                elif isinstance(result_dict["smiles"], list):
                    ref_smiles = result_dict["smiles"][0]
            gen_smiles_list = result_dict["result"]
            
            canonical_smiles = canonicalized_smiles(ref_smiles)
            gen_smiles_list = [canonicalized_smiles(_) for _ in gen_smiles_list]

            label_list.append(canonical_smiles)
            predict_list_list.append(gen_smiles_list)
        
        for topk in [1,3,5,10,50,100]:
            topk_accuracy, right, total = get_topk_accuracy(label_list, predict_list_list, topk)
            logger.info("topk:%d, accuracy:%.3f"%(topk, topk_accuracy))

    
    if local_rank != -1:
        dist.barrier()
    
    
def main(args):
    best_score = -np.inf #-∞
    local_rank = -1
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) train worker starting...")
    except Exception as e:
        print(e)
        local_rank = -1  
    print("local_rank", local_rank)
    #trainer_config
    if local_rank <= 0:
        print("args:", vars(args))
        trainer_config = args.__dict__.copy()
        with open(args.config_json_path, "r") as f:
            trainer_config["config_json_context"] = json.loads(f.read())
        trainer_config["lr_scheduler_type"] = str(trainer_config["lr_scheduler_type"])
        trainer_config["location"] = str(os.uname())
        with open(os.path.join(args.output_dir, "config.json"), "w") as f:
            json.dump(trainer_config, f, indent=4)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if local_rank!=-1:
        dist.init_process_group(backend=args.backend, init_method='env://', timeout=datetime.timedelta(0, 14400))
        torch.cuda.set_device(local_rank)
        torch.backends.cudnn.benchmark = True
        model_config = {
                        "config_json_path": args.config_json_path,
                        "tokenizer_path": args.tokenizer_path
                        }
        model = MultiConstraintMolecularGenerator(**model_config)
        # args.model_weight = None
        if (args.model_weight is not None) and os.path.exists(args.model_weight):
            
            model.load_weights(args.model_weight) 
            if local_rank <= 0:
                print("load weights")
        model = model.to(device) 
        tokenizer = model.tokenizer

        
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)    
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        args.gpus = dist.get_world_size()
    else:
        model_config = {"model_path": args.model_path,
                        "config_json_path": args.config_json_path,
                        "tokenizer_path": args.tokenizer_path,
                        }
        model = MultiConstraintMolecularGenerator(**model_config)
        if (args.model_weight is not None) and os.path.exists(args.model_weight):
            model.load_weights(args.model_weight) 
            print("load weights")
        model = model.to(device)
        tokenizer = model.tokenizer
        
    if local_rank <= 0:
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        hterm = logging.StreamHandler()
        hterm.setLevel(logging.ERROR)
        hfile = logging.FileHandler(os.path.join(args.output_dir, 'log.log'))
        hfile.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        hterm.setFormatter(formatter)
        hfile.setFormatter(formatter)
        logger.addHandler(hterm)
        logger.addHandler(hfile)
        logger.info(vars(args))
    kwargs = {
                "length_penalty": args.length_penalty,
                "num_beams": args.num_beams,
            }
    
    if args.do_train:
        train_dataset = MyDataset(args, tokenizer, args.train_folder, 
                                max_length = args.max_length, 
                                input_name = args.input_name, 
                                output_name = args.output_name,
                                phase="train")
        
        if args.validation_folder is not None:
            valid_dataset = MyDataset(args, tokenizer, args.validation_folder, 
                                    max_length = args.max_length, 
                                    input_name = args.input_name, 
                                    output_name = args.output_name,
                                    phase="val")
        if local_rank != -1: 
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            if args.validation_folder is not None:
                valid_sampler = DistributedSampler(valid_dataset, shuffle=False)
            # valid_sampler = SequentialSampler(valid_dataset)
        else:
            train_sampler = RandomSampler(train_dataset)
            if args.validation_folder is not None:
                valid_sampler = SequentialSampler(valid_dataset)
            
        if local_rank!=-1: 
            train_dataloader = DataLoader(train_dataset, 
                                        sampler=train_sampler, 
                                        batch_size=args.batch_size, ## batch_size
                                        num_workers=args.num_workers, ## num_workers
                                        # prefetch_factor=args.num_workers//2,
                                        # persistent_workers=True,
                                        pin_memory=True,
                                        drop_last=False,
                                        collate_fn=train_dataset.collate_fn)
            if args.validation_folder is not None:
                valid_dataloader = DataLoader(valid_dataset,
                                        sampler=valid_sampler,
                                        batch_size=args.batch_size*2,
                                        num_workers=args.num_workers, 
                                        # prefetch_factor=args.num_workers//2,
                                        # persistent_workers=True,
                                        pin_memory=True,
                                        drop_last=False,
                                        collate_fn=valid_dataset.collate_fn)
        else:
            train_dataloader = DataLoader(train_dataset,
                                        sampler=train_sampler,
                                        batch_size=args.batch_size, 
                                        num_workers=args.num_workers,
                                        persistent_workers=False,
                                        pin_memory=False,
                                        drop_last=False,
                                        collate_fn=train_dataset.collate_fn)
            if args.validation_folder is not None:
                valid_dataloader = DataLoader(valid_dataset,
                                        sampler=valid_sampler,
                                        batch_size=args.batch_size*2, 
                                        num_workers=args.num_workers,
                                        persistent_workers=False,
                                        pin_memory=False,
                                        drop_last=False,
                                        collate_fn=valid_dataset.collate_fn)
        
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        
        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(
                args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_epochs * len(train_dataloader),
            num_training_steps=args.max_train_steps,
        )
        start_epoch = 0

        
        total_batch_size = args.gpus * args.batch_size
        if local_rank <= 0:
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {len(train_dataset)}")
            logger.info(f"  Num Epochs = {args.num_train_epochs}")
            logger.info(
                f"  Instantaneous batch size per device = {args.batch_size}")
            logger.info(
                f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
            logger.info(
                f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {args.max_train_steps}")
            logger.info(f"======Start Training=======")
        
        completed_steps = 0
        model.train()
        scaler = torch.cuda.amp.GradScaler(enabled=args.fp32)
        for epoch in range(start_epoch, args.num_train_epochs):
            if local_rank != -1:
                train_sampler.set_epoch(epoch)
                dist.barrier()
            
            # train_score  = {
            #     "accurancy": 0,
            #     "valid": 0,
            #     "tanimoto_sim":0,
            #     "count":0,
            # }
            
            valid_score  = {
                "accurancy": 0,
                "valid": 0,
                "tanimoto_sim":0,
                "count":0,
            }
            
            model.train()
            train_loss_sum = 0.0
            start = time.time()
            for step, (idx_list, smiles_list, batch) in enumerate(train_dataloader):
                for k,v in batch.items():
                    batch[k] = v.to(device)
                
                with torch.cuda.amp.autocast(enabled=args.fp32):
  
                    outputs = model(**batch) #odict_keys(['loss', 'logits', 'encoder_last_hidden_state'])
                    losses = outputs.loss

                losses = losses / args.gradient_accumulation_steps #/args.gpus #dist.get_world_size() 
                scaler.scale(losses).backward()
                # losses.backward()
                train_loss_sum += losses.item()
                
                # pred_smiles = get_smiles(tokenizer, outputs)
                # train_pred_smiles_list.extend(pred_smiles)
                # train_origin_smiles_list.extend(smiles_list)
                
                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    scaler.unscale_(optimizer)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    # progress_bar.update(1)
                    completed_steps += 1
                
                if completed_steps >= args.max_train_steps:
                    break
                
                print_step = max(1, min(len(train_dataloader) // 20, 500))
                if (local_rank<=0) and (step + 1) % print_step == 0:
                    str_1 = "Epoch {:04d} | Step {:04d}/{:04d} | Loss {:.4f} | Time {:.4f}".format(
                        epoch + 1, step + 1, len(train_dataloader),
                        train_loss_sum / (step + 1), time.time() - start)
                    str_2 = "Learning rate = {}".format(
                        optimizer.state_dict()['param_groups'][0]['lr'])
                    logger.info(str_1 + "\n" + str_2)
            
            # train_score = get_score(train_pred_smiles_list, train_origin_smiles_list)
        
            if args.validation_folder is not None:
                model.eval()
                losses = []
                valid_pred_smiles_list = []
                valid_origin_smiles_list = []
                
                for step, (idx_list, smiles_list, batch) in enumerate(valid_dataloader):
                    for k,v in batch.items():
                        batch[k] = v.to(device)
                    
                    with torch.cuda.amp.autocast(enabled=args.fp32):
                        
                        if "labels" in batch:
                            del batch["labels"]
                        if "decoder_input_ids" in batch:
                            decoder_input_ids = batch.pop("decoder_input_ids")
                        batch['num_beams'] = 1
                        
                        with torch.no_grad():
                            if local_rank >= 0: ## 多进程
                                result = model.module.infer_2(**batch)["smiles"] # List
                            else:
                                result = model.infer_2(**batch)["smiles"] # List
                                
                    # pred_smiles = get_smiles(tokenizer, outputs)
                    pred_smiles = result
                    torch.cuda.empty_cache()
                    
                    valid_pred_smiles_list.extend(pred_smiles)
                    valid_origin_smiles_list.extend(smiles_list)
                valid_score = get_score(valid_pred_smiles_list, valid_origin_smiles_list)
                

                if local_rank >= 0: 
                    gathered_losses = [None for i in range(dist.get_world_size())]
                    dist.all_gather_object(gathered_losses, losses)
                    losses = np.mean(gathered_losses)
                    
                    
                    # gathered_train_score_list = [None for i in range(dist.get_world_size())]
                    # dist.all_gather_object(gathered_train_score_list, train_score)
                    # train_score = get_dist_score(gathered_train_score_list)
                    gathered_valid_score_list = [None for i in range(dist.get_world_size())]
                    dist.all_gather_object(gathered_valid_score_list, valid_score)
                    valid_score = get_dist_score(gathered_valid_score_list)
                    
                else: 
                    losses = np.mean(losses)
                if local_rank <= 0:    
                    try:
                        perplexity = math.exp(losses)
                    except OverflowError:
                        perplexity = float("inf")

                    logger.info(f"epoch {epoch+1}: perplexity: {perplexity}")
                    # logger.info("train_score:%s"%(json.dumps(train_score)))
                    logger.info("valid_score:%s"%(json.dumps(valid_score)))
            

            if local_rank <= 0:                
                checkpoint_dict = {
                    "epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": lr_scheduler.state_dict(),
                }
                

                if (epoch+1) % 10 == 0:
                    name = "epoch_{}_loss_{}.pth".format(epoch+1, round(train_loss_sum / len(train_dataloader), 6))
                    torch.save(model.state_dict(), os.path.join(args.output_dir, name))
                    torch.save(
                        checkpoint_dict,
                        os.path.join(args.output_dir, name.replace(".pth", ".checkpoint")))
                
                if args.validation_folder is not None:
                    if best_score < valid_score["accurancy"]:
                        logger.info(f'save best model! best score={valid_score["accurancy"]}')
                        best_score = valid_score["accurancy"]
                        name = "best.pth"
                        torch.save(model.state_dict(), os.path.join(args.output_dir, name))
                        torch.save(
                            checkpoint_dict,
                            os.path.join(args.output_dir, name.replace(".pth", ".checkpoint")))
            
            if local_rank != -1:
                dist.barrier()

    if args.do_test:
        
        if args.test_folder is not None and (os.path.exists(args.test_folder)):
            test_dataset = MyDataset(args, tokenizer, args.test_folder, 
                                max_length = args.max_length,
                                input_name=args.input_name, 
                                output_name=args.output_name,
                                phase="test")
            if local_rank != -1: 
                test_sampler = DistributedSampler(test_dataset, shuffle=False)
            else:
                test_sampler = SequentialSampler(test_dataset)
            
            if args.num_beams==1:
                test_batch_size = 128
            elif args.num_beams==100:
                test_batch_size = 2
            else:
                test_batch_size = 16
            if local_rank!=-1: 
                # import ipdb
                # ipdb.set_trace()
                test_dataloader = DataLoader(test_dataset,
                                        sampler=test_sampler,
                                        batch_size=test_batch_size, 
                                        num_workers=args.num_workers, 
                                        # prefetch_factor=args.num_workers,
                                        # persistent_workers=True,
                                        pin_memory=True,
                                        drop_last=False,
                                        collate_fn=test_dataset.collate_fn)
            else:
                test_dataloader = DataLoader(test_dataset,
                                        sampler=test_sampler,
                                        batch_size=test_batch_size, 
                                        num_workers=args.num_workers,
                                        persistent_workers=False,
                                        pin_memory=False,
                                        drop_last=False,
                                        collate_fn=test_dataset.collate_fn)
            
            if args.do_test:
                test(args, model, test_dataloader, kwargs, local_rank, tokenizer, device, use_best_model=False)


if __name__ == "__main__":
    args = parse_args()
    main(args)
