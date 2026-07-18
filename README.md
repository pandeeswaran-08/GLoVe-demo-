# GloVe Word Embeddings Demo

A simple Python project to load pretrained GloVe (Global Vectors for Word Representation) embeddings and explore word similarity and analogy relationships (e.g. `king - man + woman ≈ queen`).

## What is GloVe?

GloVe is a word embedding technique that learns vector representations of words by factorizing a global word co-occurrence matrix built from a large corpus. Unlike purely predictive models (like Word2Vec) that only look at local context windows, GloVe captures corpus-wide statistical relationships between words, producing vectors where linear directions encode semantic relationships.

## Project structure

```
glove-demo/
├── main.py          # Loads vectors, runs similarity + analogy demos
├── README.md         # This file
└── .venv/            # Virtual environment (not committed)
```

## Requirements

- Python 3.10+
- numpy

Install dependencies:

```bash
pip install numpy --break-system-packages
```

## Vector file

This project uses a pretrained vector file:

```
wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt
```

- 50-dimensional vectors
- Trained on a combined Wikipedia + Gigaword corpus
- Format: `word v1 v2 v3 ... v50` per line (space-separated, plain text)

Place the file anywhere and update the `glove_path` variable in `main.py` to point to it.

## Usage

Run the script:

```bash
python main.py
```

Expected output:

```
Loading GloVe vectors...
Loaded 400000 words in 12.3s
[('prince', 0.82), ('monarch', 0.79), ('throne', 0.77), ...]
[('queen', 0.86), ('princess', 0.78), ('monarch', 0.75)]
```

## How it works

1. **Load** — reads the vector file line by line into a word list and a numpy matrix.
2. **Normalize** — all vectors are L2-normalized once, so cosine similarity becomes a simple dot product.
3. **Similarity** — `most_similar(word)` computes `normalized_matrix @ normalized_vector` in one vectorized operation (fast, avoids per-word Python loops).
4. **Analogy** — `analogy(a, b, c)` computes `vector(a) - vector(b) + vector(c)` and finds the closest matching word, e.g. `king - man + woman ≈ queen`.

## Performance notes

- Loading time depends on vocabulary size (can take 10–60s for large files).
- Similarity/analogy lookups are vectorized with numpy, so they run in milliseconds even on large vocabularies — avoid pure Python loops over all words, as they scale poorly on machines with limited RAM (e.g. 8GB).

## Next steps

- Compare against a Word2Vec model trained on the same corpus.
- Visualize embeddings in 2D using PCA or t-SNE.
- Swap in higher-dimensional vectors (100d/300d) for richer relationships, at the cost of more RAM.

## License

For educational and personal project use.
