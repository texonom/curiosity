import time
from typing import Dict
import json
import asyncio
import aiohttp

import fire
import numpy as np
import chromadb
from datasets import load_dataset, Dataset
from tei import TEIClient
from huggingface_hub import HfApi
import vecs
import faiss as vdb

from curiosity.data import load_documents


def pgvector(dataset_id="texonom/texonom-md", dimension=384,
             prefix="", subset=None, stream=False, pgstring=None,
             tei_host="localhost", tei_port='8080', tei_protocol="http",
             batch_size=1000, start_index=None, end_index=None):
  # Load DB and dataset
  assert pgstring is not None
  vx = vecs.create_client(pgstring)
  docs = vx.get_or_create_collection(name="texonom-md", dimension=dimension)
  dataset = load_dataset(dataset_id, subset, streaming=stream)['train']

  # Filter dataset
  if not stream and end_index is not None:
    dataset = dataset[:int(end_index)]
    dataset = Dataset.from_dict(dataset)
  if not stream and start_index is not None:
    dataset = dataset[int(start_index):]
    dataset = Dataset.from_dict(dataset)

  # Batch processing function
  teiclient = TEIClient(host=tei_host, port=tei_port, protocol=tei_protocol)

  def batch_encode(batch_data: Dict) -> Dict:
    start = time.time()
    batch_zip = zip(batch_data['id'], batch_data['title'],
                    batch_data['refs'], batch_data['text'], batch_data['parent'], batch_data['created'],
                    batch_data['edited'], batch_data['creator'], batch_data['editor'])
    rows = [{'id': row[0], 'title': row[1], 'refs': row[2], 'text': row[3], 'parent': row[4],
             'created': row[5], 'edited': row[6], 'creator': row[7], 'editor': row[8]}
            for row in batch_zip]
    input_texts = [
        f"{prefix}{row['title']}\n{row['text']}\n{row['refs']}\nParent: {row['parent']}" for row in rows]
    embeddings = teiclient.embed_batch_sync(input_texts)
    metadatas = [{'title': row['title'] if row['title'] is not None else '',
                  'created': row['created'] if row['created'] is not None else '',
                  'edited': row['edited'] if row['edited'] is not None else '',
                  'creator': row['creator'] if row['creator'] is not None else '',
                  'editor': row['editor'] if row['editor'] is not None else ''} for row in rows]
    docs.upsert(records=[
        (row['id'], embeddings[i], metadatas[i]) for i, row in enumerate(rows)
    ])
    print(
        f"Batched {len(batch_data['id'])}rows takes ({time.time() - start:.2f}s)")
    return {'embeddings': embeddings, 'query': input_texts}

  # Batch processing
  dataset.map(batch_encode, batched=True, batch_size=batch_size)
  docs.create_index()


def faiss(dataset_id="texonom/texonom-md",
          model_id="thenlper/gte-small", user="texonom",
          prefix="", subset=None, token=None, stream=False,
          tei_host="localhost", tei_port='8080', tei_protocol="http",
          faiss_path="faiss", batch_size=1000, start_index=None, end_index=None):
  dataset = load_dataset(dataset_id, subset, streaming=stream)['train']

  # Filter dataset
  if not stream and end_index is not None:
    dataset = dataset[:int(end_index)]
    dataset = Dataset.from_dict(dataset)
  if not stream and start_index is not None:
    dataset = dataset[int(start_index):]
    dataset = Dataset.from_dict(dataset)

  # Batch processing function
  teiclient = TEIClient(host=tei_host, port=tei_port, protocol=tei_protocol)
  total_embeddings = []
  total_ids = []

  def batch_encode(batch_data: Dict) -> Dict:
    start = time.time()
    batch_zip = zip(batch_data['id'], batch_data['title'],
                    batch_data['refs'], batch_data['text'], batch_data['parent'], batch_data['created'],
                    batch_data['edited'], batch_data['creator'], batch_data['editor'])
    rows = [{'id': row[0], 'title': row[1], 'refs': row[2], 'text': row[3], 'parent': row[4],
             'created': row[5], 'edited': row[6], 'creator': row[7], 'editor': row[8]}
            for row in batch_zip]
    input_texts = [
        f"{prefix}{row['title']}\n{row['text']}\n{row['refs']}\nParent: {row['parent']}" for row in rows]
    embeddings = teiclient.embed_batch_sync(input_texts)
    total_embeddings.extend(embeddings)
    total_ids.extend(batch_data['id'])
    print()
    print()
    print()
    print(
        f"Batched {len(batch_data['id'])}rows takes ({time.time() - start:.2f}s)")
    return {'embeddings': embeddings, 'query': input_texts}

  # Batch processing
  dataset.map(batch_encode, batched=True, batch_size=batch_size)

  index = vdb.IndexHNSWFlat(len(total_embeddings[0]), 512)
  index.hnsw.efConstruction = 200
  index.hnsw.efSearch = 128
  embeddings = np.array([np.array(embedding)
                        for embedding in total_embeddings])
  index.add(embeddings, len(total_embeddings[0]))
  with open(f"{faiss_path}/faiss.ids", 'w', encoding='utf-8') as f:
    f.write('\n'.join(total_ids))
  vdb. write_index(index, f"{faiss_path}/faiss.index")

  # Upload to Huggingface Hub
  if token is not None:
    api = HfApi(token=token)
    api.create_repo(f"{user}/md-faiss-{model_id.split('/')[1]}",
                    repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=f'{faiss_path}',
        repo_id=f"{user}/md-faiss-{model_id.split('/')[1]}",
        repo_type="dataset",
    )


def chroma(dataset_id="texonom/texonom-md",
           model_id="thenlper/gte-small", user="texonom",
           prefix="", subset=None, token=None, stream=False,
           chroma_host="localhost", chroma_port='8888',
           tei_host="localhost", tei_port='8080', tei_protocol="http",
           chroma_path="chroma", batch_size=1000, start_index=None, end_index=None):
  # Load DB and dataset
  db = chromadb.HttpClient(chroma_host, chroma_port)
  collection = db.get_or_create_collection('texonom-md')
  dataset = load_dataset(dataset_id, subset, streaming=stream)['train']

  # Filter dataset
  if not stream and end_index is not None:
    dataset = dataset[:int(end_index)]
    dataset = Dataset.from_dict(dataset)
  if not stream and start_index is not None:
    dataset = dataset[int(start_index):]
    dataset = Dataset.from_dict(dataset)

  # Batch processing function
  teiclient = TEIClient(host=tei_host, port=tei_port, protocol=tei_protocol)

  def batch_encode(batch_data: Dict) -> Dict:
    start = time.time()
    batch_zip = zip(batch_data['id'], batch_data['title'],
                    batch_data['refs'], batch_data['text'], batch_data['parent'], batch_data['created'],
                    batch_data['edited'], batch_data['creator'], batch_data['editor'])
    rows = [{'id': row[0], 'title': row[1], 'refs': row[2], 'text': row[3], 'parent': row[4],
             'created': row[5], 'edited': row[6], 'creator': row[7], 'editor': row[8]}
            for row in batch_zip]
    input_texts = [
        f"{prefix}{row['title']}\n{row['text']}\n{row['refs']}\nParent: {row['parent']}" for row in rows]
    embeddings = teiclient.embed_batch_sync(input_texts)
    metadatas = [{'title': row['title'] if row['title'] is not None else '',
                  'created': row['created'] if row['created'] is not None else '',
                  'edited': row['edited'] if row['edited'] is not None else '',
                  'creator': row['creator'] if row['creator'] is not None else '',
                  'editor': row['editor'] if row['editor'] is not None else ''} for row in rows]
    collection.upsert(ids=batch_data['id'], embeddings=embeddings,
                      documents=batch_data['text'], metadatas=metadatas)
    print(
        f"Batched {len(batch_data['id'])}rows takes ({time.time() - start:.2f}s)")
    return {'embeddings': embeddings, 'query': input_texts}

  # Batch processing
  dataset.map(batch_encode, batched=True, batch_size=batch_size)

  # Upload to Huggingface Hub
  if token is not None:
    api = HfApi(token=token)
    api.create_repo(f"{user}/md-chroma-{model_id.split('/')[1]}",
                    repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=f'{chroma_path}',
        repo_id=f"{user}/md-chroma-{model_id.split('/')[1]}",
        repo_type="dataset",
    )


def dataset(path='texonom-md', token=None):
  documents = load_documents(path)
  # for ignore root page that has limited property
  dataset = Dataset.from_list(documents[1:])
  print(f'Properteis: {dataset.column_names}')

  # Upload to Huggingface Hub
  if token is not None:
    dataset.push_to_hub(f'texonom/{path}', token=token)


if __name__ == '__main__':
  fire.Fire()
