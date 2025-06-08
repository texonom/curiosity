# Curiosity

Our civilization is built on curiosity. Curiosity recommender system's object is suggesting perfect list after reading documents.

## Processing

1. Notion.so raw data generation
2. Nosion.so raw data to markdown

1~2 processings are done by [`texonom/notion-node`](https://github.com/texonom/notion-node)

3. Markdown to Huggingface dataset

```sh
git clone https://github.com/texonom/texonom-md
python hf_upload.py chroma
```

4. Extracted dataset to embedding

Run chroma server

```sh
pm2 start conf/chroma.json
```

Run embedding server

```sh
volume=data
model=thenlper/gte-small
docker run -d --name tei --gpus all -p 8080:80 -v $volume:/data --pull always ghcr.io/huggingface/text-embeddings-inference:0.3.0 --model-id $model
```

```bash
python index_to.py pgvector --pgstring <PGSTRING>
# or for local onnx inference
python index_to.py pgvector --pgstring <PGSTRING> --local
```

5. Use embedding for recommendation



## Plan
- [ ] from dictionary dataset without id duplicating (prefer recent one)
- [ ] dataset tagging with date 