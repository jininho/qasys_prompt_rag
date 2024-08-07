import os
import json
import weaviate
from langchain_community.document_loaders import DirectoryLoader,WebBaseLoader
import pandas as pd 

print('Start batch!')

os.environ['http_proxy'] = 'http://localhost:8080'
# 定义client对象
client = weaviate.Client(url="http://localhost:8080")

class_name = 'Stephen_Chow'
class_obj = {
    'class':class_name,
    'vectorIndexConfig':{
        'distance': 'l2-squared',   #distance是选择向量的检索方式
    },
}

# 删除以前的class
client.schema.delete_class(class_name='Stephen_Chow')
# 使用class_obj方法，创建class
client.schema.create_class(class_obj)

print('create client success!')

# 导入数据
df = pd.read_csv(r'data.csv')

# 转成list形式
sentence_data = df.sentence.tolist()

from sentence_transformers import SentenceTransformer
# 定义embeddings模型
model = SentenceTransformer('GanymedeNil/text2vec-large-chinese')


# 句子向量化
sentence_embeddings = model.encode(sentence_data)

# 将句子和embeddings后的数据整合到DataFrame里面
data = {
    'sentence': sentence_data,
    'embeddings': sentence_embeddings.tolist()
}
df = pd.DataFrame(data)

# 将数据导入Weaviate
with client.batch(
    batch_size=100
) as batch:
    for i in range(df.shape[0]):
        # print('importing data:{}'.format(i+1))
        properties = {
            'sentence_id':i+1,
            'sentence':df.sentence[i],
        }
        custom_vector = df.embeddings[i]
        client.batch.add_data_object(
            properties,
            class_name=class_name,
            vector=custom_vector
        )
print('import completed')

query = model.encode(['除暴安良'])[0].tolist()   # 这里将问题进行embeddings
nearVector = {
    'vector': query
}

response = (
    client.query
    .get(class_name, ['sentence_id', 'sentence']) # 第一个参数为class名字，第二个参数为需要显示的信息
    .with_near_vector(nearVector)             # 使用向量检索，nearVector为输入问题的向量形式
    .with_limit(5)                            # 返回个数(TopK)，这里选择返回5个
    .with_additional(['distance'])            # 选择是否输出距离
    .do()
)

# 输出结果
for i in response['data']['Get'][class_name]:
    print('='*20)
    print(i['sentence'])
