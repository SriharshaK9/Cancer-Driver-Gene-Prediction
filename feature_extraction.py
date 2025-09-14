import numpy as np

# Properties for nucleotides
purine = {'A': 1, 'T': 0, 'G': 1, 'C': 0}
h_bonds = {'A': 2, 'T': 2, 'G': 3, 'C': 3}
chem_type = {'A': 1, 'T': 0, 'G': 0, 'C': 1}

properties = [purine, h_bonds, chem_type]

def extract_features_from_sequence(sequence, max_len=50):
    seq = sequence.upper()
    
    # Pad or truncate sequence
    if len(seq) < max_len:
        seq = seq + ('N' * (max_len - len(seq)))  # unknowns → 0
    else:
        seq = seq[:max_len]
    
    feature_vector = []
    for prop in properties:
        for i in range(len(seq)-1):
            n1, n2 = seq[i], seq[i+1]
            val1 = prop.get(n1, 0.0)
            val2 = prop.get(n2, 0.0)
            feature_vector.append(val1 - val2)
    
    return np.array(feature_vector)
