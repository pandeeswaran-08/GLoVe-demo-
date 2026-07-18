import numpy as np
import time

glove_path = r"C:\Users\pandeeswaran p\Desktop\wiki_giga_2024_50_MFT20_vectors_seed_123_alpha_0.75_eta_0.075_combined.txt"

def load_glove(path, dim=50):
    words = []
    vectors = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.rstrip().split(' ')
            if len(values) <= dim:
                continue  # skip header or malformed lines
            word = ' '.join(values[:-dim])
            vec = values[-dim:]
            try:
                vectors.append(np.asarray(vec, dtype='float32'))
                words.append(word)
            except ValueError:
                continue  # skip unparseable lines
    matrix = np.array(vectors)
    # normalize once so cosine similarity = simple dot product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    normalized = matrix / norms
    word_to_idx = {w: i for i, w in enumerate(words)}
    return words, matrix, normalized, word_to_idx

print("Loading GloVe vectors...")
t0 = time.time()
words, matrix, normalized, word_to_idx = load_glove(glove_path)
print(f"Loaded {len(words)} words in {time.time()-t0:.1f}s")

def most_similar(word, topn=5):
    if word not in word_to_idx:
        return f"'{word}' not in vocabulary"
    idx = word_to_idx[word]
    sims = normalized @ normalized[idx]   # vectorized cosine sim (fast!)
    best = np.argsort(-sims)[1:topn+1]    # skip itself
    return [(words[i], float(sims[i])) for i in best]

def analogy(a, b, c, topn=3):
    vec = matrix[word_to_idx[a]] - matrix[word_to_idx[b]] + matrix[word_to_idx[c]]
    vec = vec / np.linalg.norm(vec)
    sims = normalized @ vec
    exclude = {word_to_idx[a], word_to_idx[b], word_to_idx[c]}
    best = np.argsort(-sims)
    result = [(words[i], float(sims[i])) for i in best if i not in exclude][:topn]
    return result

print(most_similar('king'))
print(analogy('king', 'man', 'woman'))