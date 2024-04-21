# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import torch
from deep_training.data_helper import ModelArguments
from transformers import HfArgumentParser, BitsAndBytesConfig, GenerationConfig
from data_utils import config_args, NN_DataHelper, get_deepspeed_config, global_args, build_messages
from module_setup import MyTransformer,BaichuanConfig,BaichuanTokenizer


deep_config = get_deepspeed_config()

if __name__ == '__main__':
    config_args['seed'] = None
    parser = HfArgumentParser((ModelArguments, ))
    (model_args, ) = parser.parse_dict(config_args, allow_extra_keys=True)

    dataHelper = NN_DataHelper(model_args)
    tokenizer, config, _,_= dataHelper.load_tokenizer_and_config(config_class_name=BaichuanConfig,
                                                                 tokenizer_class_name=BaichuanTokenizer)
    config.pad_token_id = config.eos_token_id

    pl_model = MyTransformer(config=config, model_args=model_args,
                             torch_dtype=torch.float16,)

    model = pl_model.get_llm_model()
    model = model.eval()
    model.requires_grad_(False)
    if not model.quantized:
        # 按需修改，目前只支持 4/8 bit 量化 ， 可以保存量化模型
        model.half().quantize(4).cuda()
        # 保存量化权重
        # model.save_pretrained('baichuan2-7b-int4',max_shard_size="3GB")
        # exit(0)
    else:
        # 已经量化
        model.half().cuda()

    text_list = ["写一个诗歌，关于冬天",
                 "晚上睡不着应该怎么办",
                 "从南京到上海的路线",
                 "登鹳雀楼->王之涣\n夜雨寄北->",
                 "Hamlet->Shakespeare\nOne Hundred Years of Solitude->",
                 ]


    for input in text_list:
        messages = build_messages(input)
        generation_config = GenerationConfig(max_new_tokens=512,
                                             eos_token_id=config.eos_token_id,
                                             pad_token_id=config.eos_token_id,
                                             do_sample=True, top_k=5, top_p=0.85, temperature=0.3,
                                             repetition_penalty=1.1,)
        response = model.chat(tokenizer, messages=messages,generation_config=generation_config )
        print('input',input)
        print('output',response)

