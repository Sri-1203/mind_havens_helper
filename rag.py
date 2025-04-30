from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.chroma import Chroma


def get_embed_fun():
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": "cpu"}
    encode_kwargs = {"normalize_embeddings": True}
    hf = HuggingFaceBgeEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)
    return hf


query_text="I self-harm, and I stop for awhile. Then when I see something sad or depressing, I automatically want to self-harm."
#Vector database
pdf_dir="./drive/MyDrive/base_raw"
anxiety_db="./base/anxiety_db"
depresssion_db="./base/depression_db"
bipolar_db="./base/bipolar_db"
sucidial_db="./base/suidice_db"

embed_func=get_embed_fun()
depression_db=Chroma(persist_directory=depresssion_db,embedding_function=embed_func)
bipolar_db=Chroma(persist_directory=bipolar_db,embedding_function=embed_func)
anxiety_db=Chroma(persist_directory=anxiety_db,embedding_function=embed_func)
suidice_db=Chroma(persist_directory=sucidial_db,embedding_function=embed_func)

existing_items = depression_db.get(include=[])  # IDs are always included by default
existing_ids = set(existing_items["ids"])
print(f"Number of existing documents in depression_DB: {len(existing_ids)}")
existing_items = bipolar_db.get(include=[])  # IDs are always included by default
existing_ids = set(existing_items["ids"])
print(f"Number of existing documents in bipolar_DB: {len(existing_ids)}")
existing_items = anxiety_db.get(include=[])  # IDs are always included by default
existing_ids = set(existing_items["ids"])
print(f"Number of existing documents in anxiety_DB: {len(existing_ids)}")
existing_items = suidice_db.get(include=[])  # IDs are always included by default
existing_ids = set(existing_items["ids"])
print(f"Number of existing documents in suidice_DB: {len(existing_ids)}")

def RAG(query_text,result):
  results=[]
  for j,i in result["probabilities"].items():
    #print(i,j)
    if j=="Depression":
      if i>=0.55:
        results.extend(depression_db.similarity_search_with_score(query_text, k=2))
      elif i>=0.2:
        results.extend(depression_db.similarity_search_with_score(query_text, k=1))
    elif j=="Suicidal":
      if i>=0.55:
        results.extend(suidice_db.similarity_search_with_score(query_text, k=2))
      elif i>=0.2:
        results.extend(suidice_db.similarity_search_with_score(query_text, k=1))
    elif j=="Anxiety":
      if i>=0.55:
        results.extend(anxiety_db.similarity_search_with_score(query_text, k=2))
      elif i>=0.2:
        results.extend(anxiety_db.similarity_search_with_score(query_text, k=1))
    elif j=="Biploar":
      if i>=0.55:
        results.extend(bipolar_db.similarity_search_with_score(query_text, k=2))
      elif i>=0.2:
        results.extend(bipolar_db.similarity_search_with_score(query_text, k=1))
  #print(results)
  return results

def format_doc(results):
    return "\n\n".join(result[0].page_content for result in results)

