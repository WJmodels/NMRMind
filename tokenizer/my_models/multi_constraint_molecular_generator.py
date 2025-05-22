
import os
import json
import torch
from torch import nn
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig

class MultiConstraintMolecularGenerator(nn.Module):
    def __init__(self, **kwargs):
        super(MultiConstraintMolecularGenerator, self).__init__()

        self.model_path = kwargs.pop(
            "model_path") if "model_path" in kwargs.keys() else None
        self.config_json_path = kwargs.pop(
            "config_json_path") if "config_json_path" in kwargs.keys() else None
        self.tokenizer_path = kwargs.pop(
            "tokenizer_path") if "tokenizer_path" in kwargs.keys() else None
        
        self.tokenizer = None
        if self.tokenizer_path is not None:
            self.tokenizer = BartTokenizer.from_pretrained(self.tokenizer_path)

        if self.model_path is not None and os.path.exists(self.model_path):
            assert self.config_json_path is None
            self.config = BartConfig.from_pretrained(self.model_path)

            self.model = BartForConditionalGeneration.from_pretrained(
                self.model_path)

        elif self.config_json_path is not None and os.path.exists(self.config_json_path):
            with open(self.config_json_path, "r") as f:
                json_dict = json.loads(f.read())
            
            ## 对齐vocal的长度
            if self.tokenizer is not None:
                ## 保留长的
                if len(self.tokenizer) > json_dict["vocab_size"]:
                    json_dict["vocab_size"] = len(self.tokenizer)
            self.config = BartConfig(**json_dict)
            
            self.model = BartForConditionalGeneration(config=self.config)

        else:
            raise "ERROR: No Model Found.\n"

    def forward(self, **kwargs):
        return self.model(**kwargs)
            

    def infer(self, **kwargs):

        tokenizer = kwargs.pop(
            "tokenizer") if "tokenizer" in kwargs.keys() else self.tokenizer
        num_beams = kwargs.pop(
            "num_beams") if "num_beams" in kwargs.keys() else 1
        num_return_sequences = kwargs.pop(
            "num_return_sequences") if "num_return_sequences" in kwargs.keys() else num_beams
        max_length = kwargs.pop(
            "max_length") if "max_length" in kwargs.keys() else 512
        bos_token_id = kwargs.pop(
            "bos_token_id") if "bos_token_id" in kwargs.keys() else 187

        with torch.no_grad():
            result = self.model.generate(max_length=max_length,
                                         num_beams=num_beams,
                                         num_return_sequences=num_return_sequences,
                                         bos_token_id=bos_token_id,
                                         pad_token_id=1,
                                         eos_token_id=188,
                                         decoder_start_token_id=bos_token_id, ## 重要，强制从187开始解码
                                         **kwargs)
        # print(result)
        dict_ = {"input_ids_tensor": result}
        if tokenizer is not None:
            smiles_list = []
            for _ in range(len(result)):
                try:
                    smiles = [tokenizer.decode(i) for i in result[_] if i<202 or i == 4034] #if i<202
                    smiles = [i.replace("<CLASS>", "").replace("</CLASS>", "").replace("<SMILES>", "").replace("</SMILES>", "").replace("<MATERIALS>", "").replace("</MATERIALS>", "").replace("</QED>", "").replace("<QED>", "").replace("<logP>", "").replace("</logP>", "").replace("<pad>", "").replace("</s>", "").replace("</fragment>", "").replace("<fragment>", "").replace("<SA>", "").replace("</SA>", "").replace("<mask>", "") for i in smiles]
                    smiles = "".join(smiles)
                    # print("       smiles",smiles)
                    smiles_list.append(smiles)
                except Exception as e:
                    # print(e)
                    smiles_list.append(None)
            dict_["smiles"] = smiles_list
            
        return dict_


    def load_weights(self, path, device=torch.device("cpu")):
        if path is not None:
            model_dict = torch.load(path, map_location=device)
            import collections
            new_model_dict = collections.OrderedDict()
            for k,v in model_dict.items():
                if k[:7] == "module.":
                    new_model_dict[k[7:]] = v
                else:
                    new_model_dict[k] = v
            try:
                # self.load_state_dict(new_model_dict, strict=False) #新增加了strict=False函数

                #逐层导入
                for name, param in self.named_parameters():

                    if name in new_model_dict:
                        if new_model_dict[name].shape == param.shape:
                            param.data.copy_(new_model_dict[name])
                            # print(f"Loaded weights for layer: {name}")
                        else:
                            print(f"no Loaded weights for layer: {name}")
                    else:
                        print(f"No weights found for layer: {name}")
                        
            except Exception as e:
                print("not strict load")
                
                new_model_dict["model.final_logits_bias"] = torch.cat([new_model_dict["model.final_logits_bias"], 
                                                        torch.randn([1, self.model.final_logits_bias.shape[1] - new_model_dict["model.final_logits_bias"].shape[1]])
                                                        ],dim=-1)
                
                new_model_dict["model.model.shared.weight"] = torch.cat([new_model_dict["model.model.shared.weight"], 
                                                                        torch.randn([self.model.model.shared.weight.shape[0] - new_model_dict["model.model.shared.weight"].shape[0], 768])
                                                                        ],dim=0)
                new_model_dict["model.model.encoder.embed_tokens.weight"] = torch.cat([new_model_dict["model.model.encoder.embed_tokens.weight"], 
                                                                                    torch.randn([self.model.model.encoder.embed_tokens.weight.shape[0] - new_model_dict["model.model.encoder.embed_tokens.weight"].shape[0], 768])
                                                                                    ],dim=0)
                
                new_model_dict["model.model.decoder.embed_tokens.weight"] = torch.cat([new_model_dict["model.model.decoder.embed_tokens.weight"], 
                                                                                        torch.randn([self.model.model.decoder.embed_tokens.weight.shape[0] - new_model_dict["model.model.decoder.embed_tokens.weight"].shape[0], 768])
                                                                                        ],dim=0)
                new_model_dict["model.lm_head.weight"] = torch.cat([new_model_dict["model.lm_head.weight"], 
                                                                                        torch.randn([self.model.lm_head.weight.shape[0] - new_model_dict["model.lm_head.weight"].shape[0], 768])
                                                                                        ],dim=0)
                self.load_state_dict(new_model_dict, strict=False)