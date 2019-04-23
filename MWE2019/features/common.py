import numpy as np
from ..materials import Materials

def get_tags(materials: Materials):
    qies = ['qie'] * len(materials.qies)
    idioms = ['idiom'] * len(materials.idioms)
    return qies + idioms

# retrieved from 
# https://github.com/RaRe-Technologies/gensim/blob/bd199aa0382bbbabcf7862ca80f65d7484a450cb/gensim/models/keyedvectors.py#L864
def cosine_similarities(vector_1, vectors_all):    
    """Compute cosine similarities between one vector and a set of other vectors.
    Parameters
    ----------
    vector_1 : numpy.ndarray
        Vector from which similarities are to be computed, expected shape (dim,).
    vectors_all : numpy.ndarray
        For each row in vectors_all, distance from vector_1 is computed, expected shape (num_vectors, dim).
    Returns
    -------
    numpy.ndarray
        Contains cosine distance between `vector_1` and each row in `vectors_all`, shape (num_vectors,).
    """
    norm = np.linalg.norm(vector_1)
    all_norms = np.linalg.norm(vectors_all, axis=1)
    dot_products = np.dot(vectors_all, vector_1)
    similarities = dot_products / (norm * all_norms)
    return similarities