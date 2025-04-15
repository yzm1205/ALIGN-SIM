import numpy as np

def compute_alpha_model(rnd_similarities):
    """
    Computes alpha_model as per the formula:
    
        α_model = 1 - (1 / (n * |D|)) * sum(sim(RND-Pairs))
    
    Args:
        rnd_similarities (array-like): A 2D array of shape (n, |D|) 
                                       where each entry [i][j] is the similarity 
                                       of the j-th random pair in the i-th sample.

    Returns:
        float: The computed alpha_model value.
    """
    rnd_similarities = np.array(rnd_similarities)
    n, D_size = rnd_similarities.shape
    alpha_model = 1 - (1 / (n * D_size)) * rnd_similarities.sum()
    return alpha_model


