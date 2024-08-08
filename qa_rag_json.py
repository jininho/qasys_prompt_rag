import os
import json
import weaviate
from langchain_community.document_loaders import DirectoryLoader,WebBaseLoader
import pandas as pd 
import datetime

###############################################################################
print('Start batch!')

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
client = weaviate.connect_to_local()

print('client connected!')

try:
    collection_definition = {
        "class": "TestArticle",
        # "properties": [
        #     {
        #         "name": "title",
        #         "dataType": ["text"],
        #     },
        #     {
        #         "name": "body",
        #         "dataType": ["text"],
        #     },
        # ],
        "vectorIndexConfig":{
            'distance': 'l2-squared',   #distance是选择向量的检索方式
        },
    }

    if client.collections.exists("TestArticle"):
        client.collections.delete("TestArticle")
    client.collections.create_from_dict(collection_definition)
    print('client class created!')


    # f = open('glaive_rag_v1.json')
    # data = json.load(f)
    # print('print second data',data[1])
    # f.close()

    with open('glaive_rag_v1.json', 'r') as file:
        data = json.load(file)
        #print('second data is:',data[1])
        print('load json file end!')
    file.close()



    ######################################################################
    #import data into weaviate db
    print('import data into weaviate db start!')

    print(datetime.datetime.now())

    sentence_data = data

    from sentence_transformers import SentenceTransformer
    # 定义embeddings模型
    #model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')   #useful, took long time
    #model = SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True) #can not use
    # model = SentenceTransformer('mixedbread-ai/mxbai-embed-2d-large-v1')
    # model = SentenceTransformer('McGill-NLP/LLM2Vec-Sheared-LLaMA-mntp-supervised') #can not use
    model = SentenceTransformer('sentence-transformers/multi-qa-distilbert-dot-v1')

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
                class_name="TestArticle",
                vector=custom_vector
            )
    print('import completed')
    print(datetime.datetime.now())



finally:
    client.close()
