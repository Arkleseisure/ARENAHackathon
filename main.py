import numpy as np
from pathlib import Path
import json
import torch
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer
import time
import heapq
import os

test_strings = ["This tree has very thick bark",
	            "This dog has a very loud bark",
                "At the roundabout turn right",
                "The justice system is always right",
                "That animal is a bat",	
                "That piece of wood is a cricket bat"
                ]
top_k = 128
top_excluded = 0
token_index = -1
test_location = 'mlp'
search_depth = 3
num_layers = 4


model_name = 'gpt2-xl'
if model_name == 'gpt2-small':
    out_dir = Path("gpt2small_token_resids")
elif model_name == 'gpt2-xl':
    out_dir = Path('gpt2xl_token_resids')
meta = json.load(open(out_dir/"meta.json"))
memmap_fname = meta["memmap_filename"]
dtype_np = np.float16 if 'float16' in meta["dtype"] else np.float32
vocab_size, n_layers, d_model = meta["vocab_size"], meta["n_layers"], meta["d_model"]
print(memmap_fname)
print(out_dir)
print("Loading residuals")
all_token_residuals = np.memmap(memmap_fname, dtype=dtype_np, mode="r", shape=(vocab_size, n_layers, d_model))
print("Residuals loaded")

# -*- coding: utf-8 -*-

def calc_length_of_double_list(length):
    return (length * (length - 1))//2

def calc_length_of_triple_list(length):
    return (length * (length - 1) * (length - 2))//6

percent_completed = []
# -----------------
# Load model + setup
# -----------------
print('Loading model')
model = HookedTransformer.from_pretrained(model_name)
print('Model loaded')
E = model.embed.W_E       # [vocab, d_model]
V, D = E.shape
device = 'cuda'


# Example target string
test_ids = []
for test_string in test_strings:
    test_id = model.to_tokens(test_string, prepend_bos=True)[0][token_index]
    print("Target ID:", test_id)
    print("Target token:", model.to_string(test_id))
    test_ids.append(test_id)

caches = []
for test_string in test_strings:
    _, cache = model.run_with_cache(test_string)
    caches.append(cache)

vecs = []
for test_id in test_ids:
    vec = E[test_id]
    vecs.append(vec)
ids = ''
E_norm = ''

def get_norm(vect, basis_vector, metric):
    if basis_vector[0] == float('inf'):
        return float('inf')
    # euclidean distance
    if metric == 'dist':
        b_norm = basis_vector/basis_vector.norm()
        dot_product = b_norm.dot(vect)
        distance = torch.sqrt(vect.norm()**2 - dot_product ** 2)
        return distance
    # cosine similarity
    elif metric == 'cos_sim':
        vect_norm = vect/vect.norm()
        basis_norm = basis_vector/basis_vector.norm()
        dot_product = vect_norm.dot(basis_norm)
        return dot_product
    else:
        raise ValueError("Unsupported metric. Currently supported metrics are 'dist' and 'cos_sim'.")

def get_best_vectors(vec, basis, depth=4, banned_vectors=[], verbose=False, metric='dist'): 
    vector_enumeration = [(i, vec, basis[i,]) for i in range(basis.shape[0])]
    
    for item in banned_vectors:
        vector_enumeration[item] = (item, torch.zeros(vec.shape), torch.fill(torch.zeros(vec.shape), float('inf')))
    if metric == 'dist':
        (index, vec, basis_v) = min(vector_enumeration, key=lambda x:get_norm(x[1], x[2], metric))
    elif metric == 'cos_sim':
        (index, vec, basis_v) = max(vector_enumeration, key=lambda x:get_norm(x[1], x[2], metric))
    else:
        raise ValueError

    b_norm = basis_v/basis_v.norm()
    dot_product = b_norm.dot(vec)
    factor = dot_product / basis_v.norm()
    if depth == 1:
        return [index], [factor], get_norm(vec, basis_v, metric)
    
    banned_vectors.append(index)
    best_vectors, factors, distance = get_best_vectors(vec-(factor * basis_v), basis, depth-1, banned_vectors, verbose=verbose)

    best_vectors.append(index)
    factors.append(factor)
    return best_vectors, factors, distance
    

def get_best_vecs(layer, layer_no, cache, depth=3, metric='dist'):
    if layer_no == -1:
        basis = E
    else:
        basis = torch.tensor(all_token_residuals[:, layer_no, :]).to(torch.float32).to(device)
        mu = basis.mean(0, keepdim=True)
        mu_v = basis.mean(0)

    print(f"{layer}:")
    target_vector = cache[layer][0][token_index]
    vecs, factors, dist = get_best_vectors(target_vector, basis, depth=depth, metric='dist')
    print(f'token breakdown depth {depth}:')
    for i in range(len(vecs)):
        print(f"{round(float(factors[i]), 2)} x", f"{model.to_string(vecs[i])}")
    print('Distance:', dist)
    print('Relative distance:', dist/target_vector.norm())


# print(get_best_vectors(test_vec, test_basis, depth=2))
for i, cache in enumerate(caches):
    print(test_strings[i])
    for layer in range(num_layers):
        get_best_vecs(f"blocks.{layer}.hook_{test_location}_out", layer, cache, depth=search_depth)
        print('\n')
