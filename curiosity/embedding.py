from typing import List, Tuple

from transformers import AutoTokenizer, AutoModel
from huggingface_hub import hf_hub_download
import onnxruntime as ort
import numpy as np


def average_pool(last_hidden_states,
                 attention_mask):
  last_hidden = last_hidden_states.masked_fill(
      ~attention_mask[..., None].bool(), 0.0)
  return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def encode_hf(input_texts: List[str], model_id: str = 'intfloat/multilingual-e5-large',
              prefix: str = 'intfloat/multilingual-e5-large'):
  import torch.nn.functional as F
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  model = AutoModel.from_pretrained(model_id)
  input_texts = [prefix + input_text for input_text in input_texts]
  # Tokenize the input texts
  batch_dict = tokenizer(input_texts, max_length=512,
                         padding=True, truncation=True, return_tensors='pt')
  outputs = model(**batch_dict)
  embeddings = average_pool(outputs.last_hidden_state,
                            batch_dict['attention_mask'])
  # normalize embeddings
  embeddings = F.normalize(embeddings)
  return embeddings


def load_onnx(model_id: str = 'texonom/multilingual-e5-small-4096') -> Tuple[AutoTokenizer, ort.InferenceSession]:
  """Load tokenizer and ONNX session for local inference."""
  tokenizer = AutoTokenizer.from_pretrained(model_id)
  onnx_path = hf_hub_download(model_id, 'onnx/model_quantized.onnx')
  session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
  return tokenizer, session


def encode_onnx(input_texts: List[str], tokenizer: AutoTokenizer,
                session: ort.InferenceSession, prefix: str = '') -> List[List[float]]:
  """Encode texts using an ONNX model."""
  input_texts = [prefix + text for text in input_texts]
  batch_dict = tokenizer(input_texts, max_length=512,
                         padding=True, truncation=True, return_tensors='np')
  ort_inputs = {k: v for k, v in batch_dict.items()}
  if 'token_type_ids' not in ort_inputs:
    ort_inputs['token_type_ids'] = np.zeros_like(batch_dict['input_ids'])
  # onnxruntime expects int64 inputs
  for key in ort_inputs:
    ort_inputs[key] = ort_inputs[key].astype('int64')
  outputs = session.run(None, ort_inputs)[0]
  attention_mask = batch_dict['attention_mask']
  masked = np.where(attention_mask[..., None] == 1, outputs, 0.0)
  summed = masked.sum(axis=1)
  counts = attention_mask.sum(axis=1, keepdims=True)
  embeddings = summed / counts
  embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
  return embeddings.tolist()
