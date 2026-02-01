# F020_weibo_gaoxiao vue+flask高校舆情分析系统带情感分析与微博信息抓取 （高校版本）

> 文章结尾部分有CSDN官方提供的学长 联系方式名片
> 
关注B站，有好处！
up主B站账号： **麦麦大数据**
编号:  F020 gaoxiao
## 视频演示

[video(video-OCdRtbat-1758268739273)(type-bilibili)(url-https://player.bilibili.com/player.html?aid=282419328)(image-https://i-blog.csdnimg.cn/img_convert/a33a72ccaa31936d86d3687001d3ba93.jpeg)(title-vue+flask 高校舆情大数据 | 情感分析| 高校舆情处理分析、大数据毕业设计)]

## 1系统简介
微博舆情情感分析系统是基于Vue、Flask、MySQL和LSTM架构开发的一款智能舆情监测和分析工具。该系统旨在实时爬取微博平台上的信息，包括热门话题和文章，并通过先进的LSTM模型进行情感分析，实现对舆情的深度理解和监控。系统不仅能够分析情感倾向（积极、消极），还支持舆情趋势的实时追踪及可视化展示，帮助用户更直观地理解舆情发展动态。通过多维度的数据分析与展示，用户可以及时掌握热点事件的情感波动，进而作出科学决策。此外，系统还提供用户管理功能，确保个性化的用户体验。
## 2 功能设计
该舆情分析系统的功能设计包括多个核心模块。首先，微博信息爬取模块负责定期抓取微博上的话题和文章数据，为后续分析提供基础数据。接着，基于BERT的三分类情感分析模块对抓取的数据进行处理，识别情感倾向。舆情趋势分析模块通过数据挖掘，展示舆情的历史变化及未来趋势。可视化分析模块则将数据以舆情旭日图、IP归属地分析和舆情散点图等形式呈现，便于用户理解。舆情监测功能允许用户订阅特定话题，实现实时跟踪和报告生成。同时，利用textrank与tfidf算法进行关键词分析，进一步提炼信息。最后，系统包括用户管理模块，以实现权限控制和个性化服务，确保用户的使用体验。
### 2.1  项目架构
本项目为全新的2025升级版本。通过对微博的热门话题的微博内容爬取，可以挖掘在微博某一个话题下的评论的舆情情况，方法就是利用深度学习情感分析技术。本文系统使用的自己训练的LSTM深度学习模型，使用了vue来结合echarts开发，对数据进行了多种类型的图形可视化。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/def45a7ba7e649dbab0871137d09aa5a.png)
### 2.2 模块图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0647c0e0b32c41ffb596ce577ad20c00.jpeg)

### 2.3 文档介绍 12900 字
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/922ac56f8d4543d49de65e772fbda2b1.png)

## 3功能介绍

登录与注册: 动态登录与注册界面
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e2088ce9c416453eab78e2f99306fec3.jpeg)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/dea8bd69a7634d809dd7cc61f97a4c3c.jpeg)

主页 : 显示微博话题卡片，显示舆情内容，可以点击查看话题下的微博内容和分析结果:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/08b910951f4f4300a29af74788c9cb1b.jpeg)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/77b9400cad694692b54638a8c0f11140.jpeg)

数据大屏：以大屏分隔展示系统数据情况，话题的热度、消极内容比例等角度进行数据分析可视化:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/19f4be4e4ea542bb882b46757cdf1cb4.jpeg)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/743d004b890145c3b9433206afe8d5d9.jpeg)

数据分析：话题舆情旭日图可查看话题的评论数量和情感分析分类情况、日历热点图不同时间段之评论之热度 
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/230398b56f11460793652938b41e8757.jpeg)


情感分析：给出评论之情感分析，话题已卡片方式来呈现可以，那就可以看情感分析了,同时在话题的卡片中，也可以查看到对应评论的情感分析  
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/056da79abd07426a8ed2dc469e2c1463.jpeg)

关键词分析：某一个话题的观点分析、某一个话题的关键词分析，利用关键词接口来做，通过tf-idf和textrank两种算法返回关键词  
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e4970c68abbd4514b954431809cbc66e.jpeg)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7f01a859e2474e78b53ce195b936a211.jpeg)

舆情分析 ：基于深度学习模型给出的情感分析结果，综合评价之后给出舆情，可以用卡片的方式给出根据或者图表，  在表格上，如果负面评价高于60%会显示红色，高于45%会显示橙色，低于45%会显示绿色  
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d009820f4a7a4e4ca612be6e697aab30.jpeg)
话题搜索 ：以数据表格的形式展示评论  
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/41fc9d7e5cb64be6bfb9c47ad0317975.jpeg)
我的： 修改用户信息等 ,还支持：
修改头像
修改密码
通过身份证实名认证功能。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4eab8b8c20dd4d89aea11f3a168e3f77.jpeg)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cc838a89b5754cd7a6e80b76d3e8b7ea.jpeg)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/acfef76498b6464dacaf65ed8cfc90ce.jpeg)

基于scrapy的微博爬虫
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/7021da0b0baa4fda83f179cdbabae583.jpeg)
爬取过程情感分析
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2bb3703df2784ba48e34c071b88cb314.jpeg)
数据预处理代码
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/6098ba1660734a809508c798ac339112.jpeg)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/e4c4e695d9b540079e47d173b8bfeac8.jpeg)
## 算法代码
### 代码介绍
代码说明：基于LSTM的微博舆情情感分析代码的说明：
数据加载与预处理：代码从CSV文件加载微博数据，假设数据包含“text”（文本内容）和“label”（情感标签）两列。通过预处理方法（如去除HTML标签、特殊字符等）对文本进行清洗。
模型加载与配置：使用Hugging Face的transformers库加载预训练的BERT模型（bert-base-chinese），并对模型进行配置以适应情感分析任务。通过Trainer和TrainingArguments设置训练参数。
模型训练与预测：将数据分为训练集和验证集，使用训练集微调BERT模型，并在验证集上evaluate模型性能。最后，使用训练好的模型对新数据进行情感预测。
### 流程图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/cd3cb970ad7349808db063f739ecb208.png)

### 程序源码
```python
import pandas as pd
import torch
from transformers import BertTokenizer, BertModel, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import re

# 数据加载与预处理
df = pd.read_csv("weibo_data.csv")
df["text"] = df["text"].apply(lambda x: re.sub(r'<.*?>', '', x))  # 去除HTML标签
df["text"] = df["text"].apply(lambda x: re.sub(r'[\W_]+', '', x))  # 去除特殊字符

# 分割数据集
train_texts, val_texts, train_labels, val_labels = train_test_split(df["text"], df["label"], random_state=42, test_size=0.2)

# 自定义数据集类
class WeiboDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 模型加载与配置
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
model = BertModel.from_pretrained('bert-base-chinese')

# 创建数据集实例
dataset = WeiboDataset(train_texts, train_labels, tokenizer, max_len=128)
val_dataset = WeiboDataset(val_texts, val_labels, tokenizer, max_len=128)

# 训练参数设置
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
)

# 模型训练与评估
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=val_dataset
)
trainer.train()

# 情感预测
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.last_hidden_state[:, 0, :]
    return torch.argmax(logits, dim=1)

```

