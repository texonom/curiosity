# Curiosity

Our civilization is built on curiosity. Curiosity recommender system's object is suggesting perfect list after reading documents.

## Processing

1. Data generation
2. Data to Markdown
   1~2 processings are done by [`texonom/notion-node`](https://github.com/texonom/notion-node)

3. Markdown to Huggingface dataset

```bash
python hf.py upload_markdown
```

4. Markdown to text extracted dataset

5. Extracted dataset to embedding

```bash
python hf.py upload_embedding
```

6. Use embedding for recommendation
