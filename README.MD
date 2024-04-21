## update
```text
    2024-04-22 简化
    2023-10-09 support accelerator trainer
    2023-10-07 support colossalai trainer
    2023-09-26 support transformers trainer
    2023-09-15 deep_training >= 0.2.2.post4
    2023-09-08 fix baichuan ptv2
    2023-09-06 整合baichuan finetuning ,  support baichuan2 finetuning
```
   

## install
- [deep_training](https://github.com/ssbuild/deep_training)
- pip install -U -r requirements.txt
- 如果无法安装， 可以切换官方源 pip install -i https://pypi.org/simple -U -r requirements.txt


```text
dev 通过一下方式安装 , 注意顺序
pip uninstall aigc_zoo
pip install -U git+https://github.com/ssbuild/aigc_zoo#egg=aigc_zoo
```


## v1 weight
- [Baichuan-13B-Base](https://huggingface.co/baichuan-inc/Baichuan-13B-Base)
- [Baichuan-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan-13B-Chat)
- [baichuan-13b-chat-int4](https://huggingface.co/ssbuild/baichuan-13b-chat-int4)

## v2 weight
- [Baichuan2-7B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat)
- [Baichuan2-7B-Base](https://huggingface.co/baichuan-inc/Baichuan2-7B-Base)
- [baichuan2-7b-chat-int4](https://huggingface.co/ssbuild/baichuan2-7b-chat-int4)
- 
- [Baichuan2-13B-Chat](https://huggingface.co/baichuan-inc/Baichuan2-13B-Chat)
- [Baichuan2-13B-Base](https://huggingface.co/baichuan-inc/Baichuan2-13B-Base)
- [baichuan2-13b-chat-int4](https://huggingface.co/ssbuild/baichuan2-13b-chat-int4)





## data sample
- [open_data 不定时更新数据集](https://github.com/ssbuild/open_data)
   
单条数据示例
```text
role one of user system function assistant
q question optional
a answer   must
```

```json
{"id": 1, "paragraph": [{"q": "从南京到上海的路线", "a": "你好，南京到上海的路线如下：\n1. 南京到上海，可以乘坐南京地铁1号线，在南京站乘坐轨道交通1号线。\n2. 南京到浦东机场，可以搭乘上海地铁1号，在陆家嘴站乘坐地铁1线，在浦东国际机场站乘坐机场快线，前往上海浦东国际机场。\n3. 上海到南京，可以换乘上海地铁2号线，从南京站换乘地铁2线，再从南京南站换乘地铁1路，然后到达上海站"}]}
{"id": 2, "paragraph": [{"q": "写一个诗歌，关于冬天", "a": "冬夜寂静冷，\n云在天边飘，\n冰封白雪上， \n寒冷像一场雪。\n \n雪花融化成冰，\n像那雪花飘洒，\n在寒冷的冬天，\n感受春天的喜悦。\n 冬日里，\n风雪渐消，\n一片寂静，\n把快乐和温暖带回家。"}]}
{"id": 3, "paragraph": [{"q": "晚上睡不着应该怎么办", "a": "晚上睡不着可能会让你感到焦虑或不舒服,但以下是一些可以帮助你入睡的方法:\n\n1. 制定规律的睡眠时间表:保持规律的睡眠时间表可以帮助你建立健康的睡眠习惯,使你更容易入睡。尽量在每天的相同时间上床,并在同一时间起床。\n2. 创造一个舒适的睡眠环境:确保睡眠环境舒适,安静,黑暗且温度适宜。可以使用舒适的床上用品,并保持房间通风。\n3. 放松身心:在睡前做些放松的活动,例如泡个热水澡,听些轻柔的音乐,阅读一些有趣的书籍等,有助于缓解紧张和焦虑,使你更容易入睡。\n4. 避免饮用含有咖啡因的饮料:咖啡因是一种刺激性物质,会影响你的睡眠质量。尽量避免在睡前饮用含有咖啡因的饮料,例如咖啡,茶和可乐。\n5. 避免在床上做与睡眠无关的事情:在床上做些与睡眠无关的事情,例如看电影,玩游戏或工作等,可能会干扰你的睡眠。\n6. 尝试呼吸技巧:深呼吸是一种放松技巧,可以帮助你缓解紧张和焦虑,使你更容易入睡。试着慢慢吸气,保持几秒钟,然后缓慢呼气。\n\n如果这些方法无法帮助你入睡,你可以考虑咨询医生或睡眠专家,寻求进一步的建议。"}]}

```

或者

```json
{"id": 1, "conversations": [{"from": "user", "value": "从南京到上海的路线"}, {"from": "assistant", "value": "你好，南京到上海的路线如下：\n1. 南京到上海，可以乘坐南京地铁1号线，在南京站乘坐轨道交通1号线。\n2. 南京到浦东机场，可以搭乘上海地铁1号，在陆家嘴站乘坐地铁1线，在浦东国际机场站乘坐机场快线，前往上海浦东国际机场。\n3. 上海到南京，可以换乘上海地铁2号线，从南京站换乘地铁2线，再从南京南站换乘地铁1路，然后到达上海站"}]}
{"id": 2, "conversations": [{"from": "user", "value": "写一个诗歌，关于冬天"}, {"from": "assistant", "value": "冬夜寂静冷，\n云在天边飘，\n冰封白雪上， \n寒冷像一场雪。\n \n雪花融化成冰，\n像那雪花飘洒，\n在寒冷的冬天，\n感受春天的喜悦。\n 冬日里，\n风雪渐消，\n一片寂静，\n把快乐和温暖带回家。"}]}
{"id": 3, "conversations": [{"from": "user", "value": "晚上睡不着应该怎么办"}, {"from": "assistant", "value": "晚上睡不着可能会让你感到焦虑或不舒服,但以下是一些可以帮助你入睡的方法:\n\n1. 制定规律的睡眠时间表:保持规律的睡眠时间表可以帮助你建立健康的睡眠习惯,使你更容易入睡。尽量在每天的相同时间上床,并在同一时间起床。\n2. 创造一个舒适的睡眠环境:确保睡眠环境舒适,安静,黑暗且温度适宜。可以使用舒适的床上用品,并保持房间通风。\n3. 放松身心:在睡前做些放松的活动,例如泡个热水澡,听些轻柔的音乐,阅读一些有趣的书籍等,有助于缓解紧张和焦虑,使你更容易入睡。\n4. 避免饮用含有咖啡因的饮料:咖啡因是一种刺激性物质,会影响你的睡眠质量。尽量避免在睡前饮用含有咖啡因的饮料,例如咖啡,茶和可乐。\n5. 避免在床上做与睡眠无关的事情:在床上做些与睡眠无关的事情,例如看电影,玩游戏或工作等,可能会干扰你的睡眠。\n6. 尝试呼吸技巧:深呼吸是一种放松技巧,可以帮助你缓解紧张和焦虑,使你更容易入睡。试着慢慢吸气,保持几秒钟,然后缓慢呼气。\n\n如果这些方法无法帮助你入睡,你可以考虑咨询医生或睡眠专家,寻求进一步的建议。"}]}
```



## infer
    # infer_finetuning.py 推理微调模型
    # infer_lora_finetuning.py 推理微调模型
    # infer_ptuning.py 推理p-tuning-v2微调模型
     python infer_finetuning.py
![infer](assets/1.png)
![infer](assets/2.png)

## training
```text
    # 制作数据
    cd scripts
    bash train_full.sh -m dataset 
    or
    bash train_lora.sh -m dataset 
    or
    bash train_ptv2.sh -m dataset 
    
    注: num_process_worker 为多进程制作数据 ， 如果数据量较大 ， 适当调大至cpu数量
    dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False, shuffle=True,mode='train',num_process_worker=0)
    
    # 全参数训练 
        bash train_full.sh -m train
        
    # lora adalora ia3 
        bash train_lora.sh -m train
        
    # ptv2
        bash train_ptv2.sh -m train
```
   
## 训练参数
[训练参数](args.MD)

## 友情链接

- [pytorch-task-example](https://github.com/ssbuild/pytorch-task-example)
- [moss_finetuning](https://github.com/ssbuild/moss_finetuning)
- [chatglm_finetuning](https://github.com/ssbuild/chatglm_finetuning)
- [chatglm2_finetuning](https://github.com/ssbuild/chatglm2_finetuning)
- [chatglm3_finetuning](https://github.com/ssbuild/chatglm3_finetuning)
- [t5_finetuning](https://github.com/ssbuild/t5_finetuning)
- [llm_finetuning](https://github.com/ssbuild/llm_finetuning)
- [llm_rlhf](https://github.com/ssbuild/llm_rlhf)
- [chatglm_rlhf](https://github.com/ssbuild/chatglm_rlhf)
- [t5_rlhf](https://github.com/ssbuild/t5_rlhf)
- [rwkv_finetuning](https://github.com/ssbuild/rwkv_finetuning)
- [baichuan_finetuning](https://github.com/ssbuild/baichuan_finetuning)
- [xverse_finetuning](https://github.com/ssbuild/xverse_finetuning)
- [internlm_finetuning](https://github.com/ssbuild/internlm_finetuning)
- [qwen_finetuning](https://github.com/ssbuild/qwen_finetuning)
- [skywork_finetuning](https://github.com/ssbuild/skywork_finetuning)
- [bluelm_finetuning](https://github.com/ssbuild/bluelm_finetuning)
- [yi_finetuning](https://github.com/ssbuild/yi_finetuning)

## 
    纯粹而干净的代码

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ssbuild/baichuan_finetuning&type=Date)](https://star-history.com/#ssbuild/baichuan_finetuning&Date)