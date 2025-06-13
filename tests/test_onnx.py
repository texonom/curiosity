import pytest
from curiosity.embedding import load_onnx, encode_onnx


def test_encode_onnx_output_shape():
    tokenizer, session = load_onnx('texonom/multilingual-e5-small-4096')
    embeddings = encode_onnx(["hello world"], tokenizer, session)
    assert isinstance(embeddings, list)
    assert len(embeddings) == 1
    assert isinstance(embeddings[0], list)
    assert len(embeddings[0]) > 0
