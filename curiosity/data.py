from typing import List, Dict
from pathlib import Path
import os
from re import split


def load_documents(folder_path: str) -> List[Dict]:
  glob = Path(folder_path).glob
  ps = list(glob("**/*.md"))
  print(f"Found {len(ps)} documents in {folder_path}")
  documents = []
  for p in ps:
    with open(p, 'r', encoding='utf-8') as file:
      document = {}

      # Get ID
      source = split(r"\\|/", str(p))[-1]
      slug = split(r" |.md", source)[-2]
      id = slug.split('-')[-1]

      # Get content
      content = file.read()
      parts = content.split('---')

      # Properties
      properties = parts[1].split('\n')
      for property in properties:
        if len(property.split(': ')) == 2:
          key, value = property.split(': ')
          document[key.lower()] = value
      document['text'] = '\n'.join(parts[2:])
      document['id'] = id
      documents.append(document)
  document_map = {document['id']: document for document in documents}
  documents = [document_map[id] for id in document_map.keys()]
  return documents
