import torch
from torch.utils.data import Dataset
import json
import random
import ipdb
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
import numpy as np
from typing import Any, Callable, Dict, List, NewType, Optional

USE_SMALL = False
USE_RDKIT = False

class MyDataset(Dataset):
    def __init__(self, 
                args, 
                tokenizer=None, 
                data_dir=None,
                dataset=[],
                max_length=2048,#512 
                input_name=["1H_NMR","13C_NMR","COSY","HMBC","molecular_formula"],
                output_name=["smiles"],
                phase="train",
                aug_nmr=True,
                debug=False):
        
        self.args = args
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.original_data = dataset
        self.max_length = max_length
        self.debug = debug
        self.phase = phase
        self.aug_nmr = aug_nmr
        self.input_name = input_name
        self.output_name = output_name
        self.get_kwargs(self.args)
        if len(self.original_data)==0:
            self.original_data = self.load_raw_data()
        # self.data = self.process_raw_data()

        if self.phase != "train":
            self.aug_smiles = False
            self.aug_nmr = False
        
        if self.aug_nmr:
            self.min_1H_NMR_index = self.tokenizer.convert_tokens_to_ids(self.min_1H_NMR)
            self.max_1H_NMR_index = self.tokenizer.convert_tokens_to_ids(self.max_1H_NMR)
            self.min_13C_NMR_index = self.tokenizer.convert_tokens_to_ids(self.min_13C_NMR)
            self.max_13C_NMR_index = self.tokenizer.convert_tokens_to_ids(self.max_13C_NMR)
        
    def get_kwargs(self, args):
        self.use_smiles_prob = getattr(args, "use_smiles_prob", 0.8)
        self.use_fragment_prob = getattr(args, "use_fragment_prob", 0.0)
        self.use_molecular_formula_prob = getattr(args, "use_molecular_formula_prob", 0.0)
        self.use_1H_NMR_prob = getattr(args, "use_1H_NMR_prob", 0.0)
        self.use_13C_NMR_prob = getattr(args, "use_13C_NMR_prob", 0.0)
        self.use_COSY_prob = getattr(args, "use_COSY_prob", 0.0)
        self.use_HMBC_prob = getattr(args, "use_HMBC_prob", 0.0)
        self.use_HH_prob = getattr(args, "use_HH_prob", 0.0)
        self.aug_smiles = getattr(args, "aug_smiles", True)
        

        self.min_1H_NMR = getattr(args, "min_1H_NMR", "H_0.00")
        self.max_1H_NMR = getattr(args, "max_1H_NMR", "H_15.00")
        self.precision_1H_NMR = getattr(args, "precision_1H_NMR", 0.01)
        self.jitter_rang_1H_NMR = getattr(args, "precision_1H_NMR", 0.2) 
        self.min_13C_NMR = getattr(args, "min_13C_NMR", "C_0.0")
        self.max_13C_NMR = getattr(args, "max_13C_NMR", "C_230.0")
        self.precision_13C_NMR = getattr(args, "precision_13C_NMR", 0.1)
        self.jitter_rang_13C_NMR = getattr(args, "precision_1H_NMR", 2.0) 
        
        ## mode
        self.mode = getattr(args, "mode", "forward")
    
    def jitter(self, jitter_range: float = 2, precision: float=2):
        jitter_value = np.random.uniform(-jitter_range, +jitter_range)
        encode_jitter_value = int(jitter_value/precision)
        return encode_jitter_value
    
    def get_aug_result(self, origin_input_list=[]):
        new_input_list = []
        for value in origin_input_list:
            
            try:
                if value>=self.min_13C_NMR_index and value<=self.max_13C_NMR_index:
                    new_value = value + self.jitter(self.jitter_rang_13C_NMR, self.precision_13C_NMR)
                    new_value = max(self.min_13C_NMR_index, new_value)
                    new_value = min(self.max_13C_NMR_index, new_value)
                    new_input_list.append(new_value)
                elif value>=self.min_1H_NMR_index and value<=self.max_1H_NMR_index:
                    new_value = value + self.jitter(self.jitter_rang_1H_NMR, self.precision_1H_NMR)
                    new_value = max(self.min_1H_NMR_index, new_value)
                    new_value = min(self.max_1H_NMR_index, new_value)
                    new_input_list.append(new_value)
                else:
                    new_input_list.append(value)
            except Exception as e:
                print(e)
                ipdb.set_trace()
        return new_input_list
        
        
    def load_raw_data(self):
        original_data = []
        with open(self.data_dir,"r") as f:
            for line in f:
                original_data.append(json.loads(line))
        return original_data
    
    def process_raw_data(self, examples):    
        result = {}
        if "molecular_formula" in examples.keys():
            result["molecular_formula"] = examples["molecular_formula"]
            result["molecular_formula_input_ids"] = [self.tokenizer.convert_tokens_to_ids("<molecular_formula>")] + \
                                                    [self.tokenizer.convert_tokens_to_ids(i) for i in examples["molecular_formula"]] + \
                                                    [self.tokenizer.convert_tokens_to_ids("</molecular_formula>")]
            result["molecular_formula_attention_mask"] = [1 for _ in range(len(result["molecular_formula_input_ids"]))]
            if self.debug:
                assert len(result["molecular_formula_input_ids"]) == len(result["molecular_formula_input_ids"])
        
        ## List(List())
        if "fragments" in examples.keys():
            result["fragments"] = examples["fragments"]
            result["fragments_input_ids"] = []
            result["fragments_attention_mask"] = []
            for item in examples["fragments"]:
                tmp = {}
                tmp["fragments_input_ids"] = [self.tokenizer.convert_tokens_to_ids("<fragment>")] + \
                                            [self.tokenizer.convert_tokens_to_ids(i) for i in item] + \
                                            [self.tokenizer.convert_tokens_to_ids("</fragment>")]
                tmp["fragments_attention_mask"] = [1 for _ in range(len(tmp["fragments_input_ids"]))]

                result["fragments_input_ids"].append(tmp["fragments_input_ids"])
                result["fragments_attention_mask"].append(tmp["fragments_attention_mask"])
                
                if self.debug:
                    assert len(tmp["fragments_input_ids"]) == len(tmp["fragments_attention_mask"])

        ## List(List())
        if "smiles" in examples.keys():
            if self.aug_smiles is True:
                result["smiles"] = examples["smiles"]
            else:

                result["smiles"] = [Chem.MolToSmiles(Chem.MolFromSmiles(examples["smiles"][0]))]
                
            result["smiles_input_ids"] = []
            result["smiles_attention_mask"] = []
            for item in result["smiles"]:
                tmp = {}
                tmp["smiles_input_ids"] = [self.tokenizer.convert_tokens_to_ids("<SMILES>")] + \
                                            [self.tokenizer.convert_tokens_to_ids(i) for i in item] + \
                                            [self.tokenizer.convert_tokens_to_ids("</SMILES>")]
                tmp["smiles_attention_mask"] = [1 for _ in range(len(tmp["smiles_input_ids"]))]

                result["smiles_input_ids"].append(tmp["smiles_input_ids"])
                result["smiles_attention_mask"].append(tmp["smiles_attention_mask"])
                
                if self.debug:
                    assert len(tmp["smiles_input_ids"]) == len(tmp["smiles_attention_mask"])
        
        if "1H_NMR" in examples.keys():
            result["1H_NMR"] = examples["1H_NMR"]
            
            if isinstance(examples["1H_NMR"], list) and len(examples["1H_NMR"])>0:
                if isinstance(examples["1H_NMR"][0], list):
                    tmp_list = []
                    for k, _ in enumerate(range(len(examples["1H_NMR"]))):
                        for __ in examples["1H_NMR"][_]:
                            tmp_list.append(self.tokenizer.convert_tokens_to_ids(__))
                        if k+1 != len(examples["1H_NMR"]):
                            tmp_list.append(self.tokenizer.convert_tokens_to_ids("|")) 

                        
                    result["1H_NMR_input_ids"] = [self.tokenizer.convert_tokens_to_ids("<1H_NMR>")] + \
                                                tmp_list + \
                                                [self.tokenizer.convert_tokens_to_ids("</1H_NMR>")]
                            
                
                else:
                    result["1H_NMR_input_ids"] = [self.tokenizer.convert_tokens_to_ids("<1H_NMR>")] + \
                                                    [self.tokenizer.convert_tokens_to_ids(i) for i in examples["1H_NMR"]] + \
                                                    [self.tokenizer.convert_tokens_to_ids("</1H_NMR>")]
                                                    
                result["1H_NMR_attention_mask"] = [1 for _ in range(len(result["1H_NMR_input_ids"]))]

                if self.debug:
                    assert len(result["1H_NMR_input_ids"]) == len(result["1H_NMR_attention_mask"])
        
        if "13C_NMR" in examples.keys():
            result["13C_NMR"] = examples["13C_NMR"]
            if isinstance(examples["13C_NMR"], list) and len(examples["13C_NMR"])>0:
                if isinstance(examples["13C_NMR"][0], list):
                    tmp_list = []
                    for k, _ in enumerate(range(len(examples["13C_NMR"]))):
                        for __ in examples["13C_NMR"][_]:
                            tmp_list.append(self.tokenizer.convert_tokens_to_ids(__))
                        
                        if k+1 != len(examples["13C_NMR"]):
                            tmp_list.append(self.tokenizer.convert_tokens_to_ids("|"))

                    result["13C_NMR_input_ids"] = [self.tokenizer.convert_tokens_to_ids("<13C_NMR>")] + \
                                                tmp_list + \
                                                [self.tokenizer.convert_tokens_to_ids("</13C_NMR>")]
                
                else:
                    result["13C_NMR_input_ids"] = [self.tokenizer.convert_tokens_to_ids("<13C_NMR>")] + \
                                                    [self.tokenizer.convert_tokens_to_ids(i) for i in examples["13C_NMR"]] + \
                                                    [self.tokenizer.convert_tokens_to_ids("</13C_NMR>")]

                result["13C_NMR_attention_mask"] = [1 for _ in range(len(result["13C_NMR_input_ids"]))]


                if self.debug:
                    assert len(result["13C_NMR_input_ids"]) == len(result["13C_NMR_attention_mask"])
                
                
        if "COSY" in examples.keys():
            result["COSY"] = examples["COSY"]
            tmp_list = []
            for k, _ in enumerate(range(len(examples["COSY"]))):
                for __ in examples["COSY"][_]:
                    tmp_list.append(self.tokenizer.convert_tokens_to_ids(__))
                if k+1 != len(examples["COSY"]):
                    tmp_list.append(self.tokenizer.convert_tokens_to_ids("|"))
            
            result["COSY_input_ids"] = [self.tokenizer.convert_tokens_to_ids("<COSY>")] + tmp_list + [self.tokenizer.convert_tokens_to_ids("</COSY>")]
            result["COSY_attention_mask"] = [1 for _ in range(len(result["COSY_input_ids"]))]
            

            if self.debug:
                assert len(result["COSY_input_ids"]) == len(result["COSY_attention_mask"])
                
                
        if "HMBC" in examples.keys():
            result["HMBC"] = examples["HMBC"]
            tmp_list = []
            for _ in range(len(examples["HMBC"])):
                for __ in examples["HMBC"][_]:
                    tmp_list.append(self.tokenizer.convert_tokens_to_ids(__))
                if k+1 != len(examples["HMBC"]):
                    tmp_list.append(self.tokenizer.convert_tokens_to_ids("|"))
            
            result["HMBC_input_ids"] = [self.tokenizer.convert_tokens_to_ids("<HMBC>")] + tmp_list + [self.tokenizer.convert_tokens_to_ids("</HMBC>")]
            result["HMBC_attention_mask"] = [1 for _ in range(len(result["HMBC_input_ids"]))]
            

            if self.debug:
                assert len(result["HMBC_input_ids"]) == len(result["HMBC_attention_mask"])
                
        if 'HH' in examples.keys():
            result["HH"] = examples["HH"]
            tmp_list = []
            for _ in range(len(examples["HH"])):
                for __ in examples["HH"][_]:
                    tmp_list.append(self.tokenizer.convert_tokens_to_ids(__))
                if k+1 != len(examples["HH"]):
                    tmp_list.append(self.tokenizer.convert_tokens_to_ids("|"))
            
            result["HH_input_ids"] = [self.tokenizer.convert_tokens_to_ids("<HH>")] + tmp_list + [self.tokenizer.convert_tokens_to_ids("</HH>")]
            result["HH_attention_mask"] = [1 for _ in range(len(result["HH_input_ids"]))]

            if self.debug:
                assert len(result["HH_input_ids"]) == len(result["HH_attention_mask"])

        return result
        
    def __len__(self):
        return len(self.original_data)

    def __getitem__(self, idx):
        return self.get_data(idx)
    
    def get_data(self, idx):
        item = self.process_raw_data(self.original_data[idx])
        collect_dict = {}
        
        smiles = None
        sub_smiles = None
        
        #smiles
        #item["smiles_input_ids"]:List[List[int]]
        if "smiles_input_ids" in item.keys():
            if ("smiles" in self.input_name) or ("smiles" in self.output_name):
                if (self.phase != "train" and self.use_smiles_prob > 0) or random.random() < self.use_smiles_prob:
                    rand_idx_smiles = random.randint(0, len(item["smiles_input_ids"])-1)
                    tmp_dict = {"input_ids": item["smiles_input_ids"][rand_idx_smiles],
                        "attention_mask": item["smiles_attention_mask"][rand_idx_smiles]}
                    collect_dict["smiles"] = tmp_dict
                    smiles = item["smiles"][0]

        
        #fragments
        #item["fragments_input_ids"]:List[List[int]]
        if "fragments_input_ids" in item.keys():
            if "fragments" in self.input_name or "fragments" in self.output_name:
                if (self.phase != "train" and self.use_fragment_prob > 0) or random.random() < self.use_fragment_prob:
                    rand_idx_fragments = random.randint(0, len(item["fragments_input_ids"])-1)
                    tmp_dict = {"input_ids": item["fragments_input_ids"][rand_idx_fragments],
                        "attention_mask": item["fragments_attention_mask"][rand_idx_fragments]}
                    
                    collect_dict["fragments"] = tmp_dict
        
        # molecular_formula
        if "molecular_formula_input_ids" in item.keys():
            if "molecular_formula" in self.input_name or "molecular_formula" in self.output_name:
                if (self.phase != "train" and self.use_molecular_formula_prob > 0) or random.random() < self.use_molecular_formula_prob:
                    tmp_dict = {"input_ids": item["molecular_formula_input_ids"],
                            "attention_mask": item["molecular_formula_attention_mask"]}
                    collect_dict["molecular_formula"] = tmp_dict

        #1H_NMR_input_ids
        if "1H_NMR_input_ids" in item.keys():
            if "1H_NMR" in self.input_name or "1H_NMR" in self.output_name:
                if (self.phase != "train" and self.use_1H_NMR_prob > 0) or random.random() < self.use_1H_NMR_prob:
                    if self.aug_nmr:
                        new_1H_NMR_input = self.get_aug_result(item["1H_NMR_input_ids"])
                        if self.debug:
                            assert item["1H_NMR_input_ids"]!=new_1H_NMR_input
                        
                        tmp_dict = {"input_ids": new_1H_NMR_input,
                            "attention_mask": item["1H_NMR_attention_mask"]}
                    else:
                        tmp_dict = {"input_ids": item["1H_NMR_input_ids"],
                            "attention_mask": item["1H_NMR_attention_mask"]}
                        
                    collect_dict["1H_NMR"] = tmp_dict

        
        #13C_NMR_input_ids
        if "13C_NMR_input_ids" in item.keys():
            if "13C_NMR" in self.input_name or "13C_NMR" in self.output_name:
                if (self.phase != "train" and self.use_13C_NMR_prob > 0) or random.random() < self.use_13C_NMR_prob:
                    
                    if self.aug_nmr:
                        new_13C_NMR_input = self.get_aug_result(item["13C_NMR_input_ids"])
                        if self.debug:
                            assert new_13C_NMR_input != item["13C_NMR_input_ids"]
                        
                        tmp_dict = {"input_ids": new_13C_NMR_input,
                            "attention_mask": item["13C_NMR_attention_mask"]}
                    else:
                        tmp_dict = {"input_ids": item["13C_NMR_input_ids"],
                            "attention_mask": item["13C_NMR_attention_mask"]}
                    
                    collect_dict["13C_NMR"] = tmp_dict
                    
        #COSY
        if "COSY_input_ids" in item.keys():
            if "COSY" in self.input_name or "COSY" in self.output_name:
                if (self.phase != "train" and self.use_COSY_prob > 0) or random.random() < self.use_COSY_prob:
                    if self.aug_nmr:
                        new_COSY_input = self.get_aug_result(item["COSY_input_ids"])
                        if self.debug:
                            assert new_COSY_input != item["COSY_input_ids"]
                        
                        tmp_dict = {"input_ids": new_COSY_input,
                            "attention_mask": item["COSY_attention_mask"]}
                    else:
                        tmp_dict = {"input_ids": item["COSY_input_ids"],
                            "attention_mask": item["COSY_attention_mask"]}
                    collect_dict["COSY"] = tmp_dict
                    
        #HMBC
        if "HMBC_input_ids" in item.keys():
            if "HMBC" in self.input_name or "HMBC" in self.output_name:
                if (self.phase != "train" and self.use_HMBC_prob > 0) or random.random() < self.use_HMBC_prob:
                    if self.aug_nmr:
                        new_HMBC_input = self.get_aug_result(item["HMBC_input_ids"])
                        if self.debug:
                            assert new_HMBC_input != item["HMBC_input_ids"]
                            
                        tmp_dict = {"input_ids": new_HMBC_input,
                            "attention_mask": item["HMBC_attention_mask"]}
                    else:
                        tmp_dict = {"input_ids": item["HMBC_input_ids"],
                            "attention_mask": item["HMBC_attention_mask"]}
                    collect_dict["HMBC"] = tmp_dict
        #HH
        if "HH_input_ids" in item.keys():
            if "HH" in self.input_name or "HH" in self.output_name:
                if (self.phase != "train" and self.use_HH_prob > 0) or random.random() < self.use_HH_prob:
                    if self.aug_nmr:
                        new_HH_input = self.get_aug_result(item["HH_input_ids"])
                        if self.debug:
                            assert new_HH_input != item["HH_input_ids"]
                            
                        tmp_dict = {"input_ids": new_HH_input,
                            "attention_mask": item["HH_attention_mask"]}
                    else:
                        tmp_dict = {"input_ids": item["HH_input_ids"],
                            "attention_mask": item["HH_attention_mask"]}
                    collect_dict["HH"] = tmp_dict
        

        if len(collect_dict) == 1:
            if "13C_NMR_input_ids" in item.keys():
                if "13C_NMR" in self.input_name or "13C_NMR" in self.output_name:
                    if (self.phase != "train") or random.random() < 1.0:
                        
                        if self.aug_nmr:
                            new_13C_NMR_input = self.get_aug_result(item["13C_NMR_input_ids"])
                            if self.debug:
                                assert new_13C_NMR_input != item["13C_NMR_input_ids"]
                            
                            tmp_dict = {"input_ids": new_13C_NMR_input,
                                "attention_mask": item["13C_NMR_attention_mask"]}
                        else:
                            tmp_dict = {"input_ids": item["13C_NMR_input_ids"],
                                "attention_mask": item["13C_NMR_attention_mask"]}
                        
                        collect_dict["13C_NMR"] = tmp_dict
                        

        if set(collect_dict.keys()) == {'smiles', 'molecular_formula'}:
            del collect_dict['molecular_formula']
            if "1H_NMR_input_ids" in item.keys():
                if "1H_NMR" in self.input_name or "1H_NMR" in self.output_name:
                    if (self.phase != "train") or random.random() < 1.0:
                        if self.aug_nmr:
                            new_1H_NMR_input = self.get_aug_result(item["1H_NMR_input_ids"])
                            if self.debug:
                                assert item["1H_NMR_input_ids"]!=new_1H_NMR_input
                            
                            tmp_dict = {"input_ids": new_1H_NMR_input,
                                "attention_mask": item["1H_NMR_attention_mask"]}
                        else:
                            tmp_dict = {"input_ids": item["1H_NMR_input_ids"],
                                "attention_mask": item["1H_NMR_attention_mask"]}
                            
                        collect_dict["1H_NMR"] = tmp_dict

        
        input = {"input_ids": [],
            "attention_mask": []}
        output = {"input_ids": [],
            "attention_mask": []}
        
        for key in self.input_name:
            if key in collect_dict:
                input["input_ids"].extend(collect_dict[key]["input_ids"])
                input["attention_mask"].extend(collect_dict[key]["attention_mask"])
        
        for key in self.output_name:
            if key in collect_dict:
                output["input_ids"].extend(collect_dict[key]["input_ids"])
                output["attention_mask"].extend(collect_dict[key]["attention_mask"])
        
        input = self.tokenizer.pad(input, return_tensors="pt")
        output = self.tokenizer.pad(output, return_tensors="pt")

        if self.mode != "forward":
            input, output = output, input
        
        input["idx"] = idx
        input["smiles"] = smiles
        input["input_ids"] = input["input_ids"]
        input["input_attention_mask"] = input["attention_mask"]
        input["output_ids"] = output["input_ids"]
        input["output_attention_mask"] = output["attention_mask"]
        
        if self.mode != "forward":
            input["smiles"] = sub_smiles
        
        return input
        

    def collate_fn(self, batch):
        input_max_length = 0
        output_max_length = 0
        
        for temp in batch:
            input_max_length = max(len(temp["input_ids"]), input_max_length)
            output_max_length = max(len(temp["output_ids"]), output_max_length)
        input_max_length = min(input_max_length, self.max_length)
        output_max_length = min(output_max_length, self.max_length)
        
        idx_list = []
        smiles_list = []
        input_ids_list = []
        input_attention_mask_list = []
        output_ids_list = []
        padding_num = self.tokenizer.convert_tokens_to_ids("<pad>")
        for _, temp in enumerate(batch):
            if "idx" in temp:
                idx_list.append(temp["idx"])
            if "smiles" in temp:
                smiles_list.append(temp["smiles"])

            temp_input_ids = temp["input_ids"][: input_max_length]
            temp_attention_mask = temp["attention_mask"][: input_max_length]
            temp_output_ids = temp["output_ids"][: output_max_length]
            if temp_input_ids.shape[-1] < input_max_length:
                temp_input_ids = torch.cat([temp_input_ids, 
                                    torch.ones(input_max_length-temp_input_ids.shape[0]).long()*padding_num])
                temp_attention_mask = torch.cat([temp_attention_mask, 
                                    torch.zeros(input_max_length-temp_attention_mask.shape[0])*0])
            
            input_ids_list.append(temp_input_ids)
            input_attention_mask_list.append(temp_attention_mask)
            
            if output_max_length > 0:
                if temp_output_ids.shape[-1] < output_max_length:
                    temp_output_ids = torch.cat([temp_output_ids, 
                                            torch.ones(output_max_length-temp_output_ids.shape[0])*padding_num])

                output_ids_list.append(temp_output_ids)
            
        input = {}
        input["input_ids"] = torch.stack(input_ids_list, dim=0).long()
        input["attention_mask"] = torch.stack(input_attention_mask_list, dim=0).long()
        
        if len(output_ids_list)>0:
            output_ids = torch.stack(output_ids_list, dim=0).long()
            input["labels"] = output_ids[:, 1:].clone()
            input["decoder_input_ids"] = output_ids[:, :-1].clone()

            
            return idx_list, smiles_list, input


    
    def get_new_smiles(self, old_mol):
        len_mol = old_mol.GetNumAtoms()
        li = []
        for index in range(len_mol):
            try:
                new_smiles = Chem.MolToSmiles(old_mol, rootedAtAtom=index)
                li.append(new_smiles)
            except:
                pass
        if len(li) >= 1:
            return random.choice(li)
        else:
            return None