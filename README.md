# 环境
- torch>=2.0.0
- transformers==4.38.1
- numpy
- pandas
- tqdm
# 数据
所有数据放在data目录下
# 模型
从huggingface下载Linq-Embed-Mistral到当前目录
# 方案
- query：将问题的question和body拼接到一起
- doc:将文章的title和abstract拼接到一起
- 使用Linq-Embed-Mistral的huggingface主页上示例代码，将query和doc转向量
- query使用余弦相似度往doc里召回出top20
# 运行
python main.py
