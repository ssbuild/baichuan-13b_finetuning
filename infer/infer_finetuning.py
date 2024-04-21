# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import typing
import torch
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from transformers import HfArgumentParser, AutoConfig, GenerationConfig
from data_utils import config_args, NN_DataHelper, get_deepspeed_config,build_messages
from module_setup import MyTransformer,BaichuanConfig,BaichuanTokenizer

deep_config = get_deepspeed_config()


if __name__ == '__main__':
    config_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(config_args, allow_extra_keys=True)

    dataHelper = NN_DataHelper(model_args)
    tokenizer, _, _,_= dataHelper.load_tokenizer_and_config(config_class_name=BaichuanConfig,
                                                            tokenizer_class_name=BaichuanTokenizer,)

    weight_dir = '../scripts/best_ckpt'
    config = BaichuanConfig.from_pretrained(weight_dir)
    pl_model = MyTransformer(config=config, model_args=model_args,torch_dtype=torch.float16,)

    # deepspeed 权重使用转换脚本命令
    # 一般根据时间排序选最新的权重文件夹
    # cd best_ckpt/last
    # python zero_to_fp32.py . ../last.ckpt

    train_weight = os.path.join(weight_dir,"last.ckpt")
    pl_model.load_sft_weight(train_weight,strict=True)

    # 保存hf权重
    # config.save_pretrained('convert/')

    # 保存sft p-tuning-v2 权重
    #  pl_model.save_sft_weight('convert/pytorch_model_sft_ptv2.bin')

    # 保存sft权重
    # pl_model.save_sft_weight('convert/pytorch_model_sft.bin')

    model = pl_model.get_llm_model()

    if not model.quantized:
        # 按需修改，目前只支持 4/8 bit 量化 ， 可以保存量化模型
        model.half().quantize(4).cuda()
    else:
        # 已经量化
        model.half().cuda()

    text_list = ["写一个诗歌，关于冬天",
                 "晚上睡不着应该怎么办",
                 "从南京到上海的路线",
                 ]


    for input in text_list:
        messages = build_messages(input)
        generation_config = GenerationConfig(eos_token_id=config.eos_token_id,
                                             pad_token_id=config.eos_token_id,
                                             max_new_tokens=512,
                                             do_sample=True, top_k=5, top_p=0.85, temperature=0.3,
                                             repetition_penalty=1.1,)
        response = model.chat(tokenizer, messages=messages,generation_config=generation_config )
        print('input',input)
        print('output',response)