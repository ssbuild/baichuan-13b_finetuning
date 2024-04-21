# @Time    : 2023/4/2 22:49
# @Author  : tk
# @FileName: infer_ptuning
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

import torch
from deep_training.data_helper import ModelArguments, DataArguments
from transformers import HfArgumentParser,AutoConfig

from data_utils import config_args, NN_DataHelper
from module_setup import MyTransformer,PromptArguments,BaichuanConfig,BaichuanTokenizer
from deep_training.zoo.utils.llm_generate import Generate

if __name__ == '__main__':
    config_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args,) = parser.parse_dict(config_args, allow_extra_keys=True)



    dataHelper = NN_DataHelper(model_args)
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(config_class_name=BaichuanConfig,
                                                              tokenizer_class_name=BaichuanTokenizer, )
    weight_dir = './best_ckpt/last'
    config = BaichuanConfig.from_pretrained(weight_dir)
    prompt_args = PromptArguments.from_pretrained(weight_dir)

    assert prompt_args.inference_mode == True

    new_num_tokens = config.vocab_size
    if config.task_specific_params is not None and config.task_specific_params.get('vocab_size', None) is not None:
        config.vocab_size = config.task_specific_params['vocab_size']

    pl_model = MyTransformer(config=config, model_args=model_args, prompt_args=prompt_args,
                             torch_dtype=torch.float16,new_num_tokens=new_num_tokens,
                             )
    # 加载sft权重
    pl_model.load_sft_weight(weight_dir)

    pl_model.eval().half().cuda()

    model = pl_model.get_llm_model()

    #基础模型精度
    model.base_model_torch_dtype = torch.half

    text_list = ["写一个诗歌，关于冬天",
                 "晚上睡不着应该怎么办",
                 "从南京到上海的路线"]
    for input in text_list:
        response = Generate.generate(model, query=input, tokenizer=tokenizer, max_new_tokens=512,
                                          eos_token_id=config.eos_token_id,pad_token_id=config.eos_token_id,
                                          do_sample=False, top_p=0.7, temperature=0.95, )
        print('input', input)
        print('output', response)