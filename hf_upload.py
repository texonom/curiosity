import time
from typing import Dict
import json
import asyncio
import aiohttp

import fire
import chromadb
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi
import nest_asyncio
from tei import TEIClient


nest_asyncio.apply()


async def chroma(dataset_id="texonom/texonom-md",
                 model_id="thenlper/gte-small", user="seonglae",
                 prefix="", subset=None, token=None, stream=False,
                 chroma_host="localhost", chroma_port='8000',
                 tei_host="localhost", tei_port='8080', tei_protocol="http",
                 chroma_path="chroma", batch_size=1000, start_index=None, end_index=None):
  # Load DB and dataset
  db = chromadb.HttpClient(chroma_host, chroma_port)
  collection = db.get_or_create_collection(dataset_id)
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
                    batch_data['url'], batch_data['text'])
    rows = [{'id': row[0], 'title': row[1], 'url': row[2], 'text': row[3]}
            for row in batch_zip]
    input_texts = [f"{prefix}{row['title']}\n{row['text']}" for row in rows]
    embeddings = teiclient.embed_batch_sync(input_texts, model_id)
    metadatas = [{'title': row['title'], 'url': row['url']} for row in rows]
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
    api.create_repo(f'{user}/chroma-{dataset_id}',
                    repo_type="dataset", exist_ok=True)
    api.upload_folder(
        folder_path=f'{chroma_path}/{dataset_id}',
        repo_id=f"{user}/md-chroma-{model_id.split('/')[1]}",
        repo_type="dataset",
    )


if __name__ == '__main__':
  fire.Fire()
