import os
import weaviate

os.environ["NO_PROXY"] = "localhost,127.0.0.1"
client = weaviate.connect_to_local()

jeopardy = client.collections.get('TestArticle')

try:

    ##basic search
    # response = jeopardy.query.fetch_objects(limit=1)

    # #hibrid search
    # from weaviate.classes.query import HybridFusion
    # query_vector = [-0.02] * 768
    # response = jeopardy.query.hybrid(
    #     query="school",
    #     # target_vector="sentence",
    #     alpha=1.25,
    #     # fusion_type=HybridFusion.RELATIVE_SCORE,
    #     # query_properties=["question"],
    #     # query_properties=["question^2", "answer"],
    #     vector=query_vector,
    #     limit=3,
    # )

    #keyword search
    from weaviate.classes.query import MetadataQuery
    response = jeopardy.query.bm25(
        query="school in South Africa",
        # query_properties=["question"],
        return_metadata=MetadataQuery(score=True),
        limit=2
    )

    for o in response.objects:
        print(o.properties)

finally:
    client.close()

