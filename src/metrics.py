from abc import ABC, abstractmethod
import numpy as np

# Optional: import torch if available for type checking
try:
    import torch
except ImportError:
    torch = None

def to_numpy(arr) -> np.ndarray:
    """
    Converts the input array (which can be a numpy array, torch tensor, or list) to a numpy array.
    """
    # Check for torch.Tensor if torch is available
    if torch is not None and isinstance(arr, torch.Tensor):
        # Detach and move to CPU if needed, then convert to numpy
        return arr.detach().cpu().numpy()
    # If it's already a numpy array, return as is
    if isinstance(arr, np.ndarray):
        return arr
    # Otherwise, try converting to a numpy array
    return np.array(arr)

class Metric(ABC):
    """
    Abstract base class for evaluation metrics.
    Subclasses must implement the compute method.
    """
    @abstractmethod
    def compute(self, vector1, vector2) -> float:
        """
        Compute the metric between two vectors.
        
        Args:
            vector1: The first vector (numpy array, torch tensor, list, etc.).
            vector2: The second vector (numpy array, torch tensor, list, etc.).
        
        Returns:
            float: The computed metric value.
        """
        pass

class CosineMetric(Metric):
    """
    Implementation of the cosine similarity metric.
    """
    def compute(self, vector1, vector2) -> float:
        # Convert inputs to numpy arrays
        vec1 = to_numpy(vector1)
        vec2 = to_numpy(vector2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

class NEDMetric(Metric):
    """
    Implementation of a normalized Euclidean distance metric.
    """
    def compute(self, vector1, vector2) -> float:
        # Convert inputs to numpy arrays
        vec1 = to_numpy(vector1)
        vec2 = to_numpy(vector2)
        
        euclidean_distance = np.linalg.norm(vec1 - vec2)
        norm_sum = np.linalg.norm(vec1) + np.linalg.norm(vec2)
        if norm_sum == 0:
            return 0.0
        return euclidean_distance / norm_sum

class EuclideanMetric(Metric):
    def compute(self, vector1, vector2) -> float:
        return np.linalg.norm(vector1 - vector2, axis=1)

def dot_product(x, y):
    return np.dot(x, y.T)

def compute_ned_distance(x, y):
    return 0.5 * np.var(x - y) / (np.var(x) + np.var(y))

def batch_NED(batch_u, batch_v):
    batch_u = np.array(batch_u)
    batch_v = np.array(batch_v)
    
    # Ensure batch_u and batch_v have the same number of elements
    assert batch_u.shape[0] == batch_v.shape[0], "The batch sizes of u and v must be the same."
    
    scores = []
    
    for u, v in zip(batch_u, batch_v):
        u = np.array(u)
        v = np.array(v)
        
        u_mean = np.mean(u)
        v_mean = np.mean(v)
        
        u_centered = u - u_mean
        v_centered = v - v_mean
        
        numerator = np.linalg.norm(u_centered - v_centered, ord=2)**2
        denominator = np.linalg.norm(u_centered, ord=2)**2 + np.linalg.norm(v_centered, ord=2)**2
        
        ned_score = 0.5 * numerator / denominator
        scores.append(ned_score)
    
    return np.array(scores)

    
def NED2(u, v):
    u = np.array(u)
    v = np.array(v)
    
    u_mean = np.mean(u)
    v_mean = np.mean(v)
    
    u_centered = u - u_mean
    v_centered = v - v_mean
    
    numerator = np.linalg.norm(u_centered - v_centered, ord=2)**2
    denominator = np.linalg.norm(u_centered, ord=2)**2 + np.linalg.norm(v_centered, ord=2)**2
    
    return 0.5 * numerator / denominator

# --- Example Usage ---
if __name__ == "__main__":
    # Example inputs: a numpy array and a torch tensor (if torch is available)
    vec_np = np.array([1.0, 2.0, 3.0])
    if torch is not None:
        vec_torch = torch.tensor([4.0, 5.0, 6.0])
    else:
        vec_torch = [4.0, 5.0, 6.0]  # fallback list

    cosine = CosineMetric()
    ned = NEDMetric()
    
    print("Cosine Similarity:", cosine.compute(vec_np, vec_torch))
    print("Normalized Euclidean Distance:", ned.compute(vec_np, vec_torch))

    # x = [20,30,2]
    # y = [1.0,2.0,3.0]
    # print("Dot Product: ", dot_product(x, y))
    # print(euclidean_distance(x, y))
    # # print(NED(x, y))
    # print("Done")