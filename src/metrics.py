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
    def cosine_similarity(self,a,b):
        """
        Takes 2 vectors a, b and returns the cosine similarity according 
            to the definition of the dot product
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)
    
    def compute(self,vector1, vector2) -> float:
        similarity_scores = []
        for i in range(len(vector1)):
            similarity_scores.append(self.cosine_similarity(
                vector1[i], vector2[i]))

        return np.mean(similarity_scores),similarity_scores

class NEDMetric(Metric):
    """
    Implementation of a normalized Euclidean distance metric.
    """
    def compute(self, vector1, vector2) -> np.ndarray:
        batch_u = np.array(vector1)
        batch_v = np.array(vector2)
        
        # Ensure batch_u and batch_v have the same number of elements
        assert batch_u.shape[0] == batch_v.shape[0], "The batch sizes of u and v must be the same."
        
        scores = []
        
        for u, v in zip(batch_u, batch_v):
            u_mean = np.mean(u)
            v_mean = np.mean(v)
            
            u_centered = u - u_mean
            v_centered = v - v_mean
            
            numerator = np.linalg.norm(u_centered - v_centered, ord=2)**2
            denominator = np.linalg.norm(u_centered, ord=2)**2 + np.linalg.norm(v_centered, ord=2)**2
            
            ned_score = 0.5 * numerator / denominator
            scores.append(ned_score)
        
        return np.mean(scores),np.array(scores)


class EuclideanMetric(Metric):
    def compute(self, vector1, vector2) -> float:
        return np.linalg.norm(vector1 - vector2, axis=1)

    
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
    # # Example inputs: a numpy array and a torch tensor (if torch is available)
    # vec_np = np.array([1.0, 2.0, 3.0])
    # if torch is not None:
    #     vec_torch = torch.tensor([4.0, 5.0, 6.0])
    # else:
    #     vec_torch = [4.0, 5.0, 6.0]  # fallback list

    # Example Usage
    vec_np = np.array([[1, 2, 3], [4, 5, 6]]) # changed to np array
    vec_torch = np.array([[7, 8, 9], [10, 11, 12]]) # changed to np array
    # cosine = CosineMetric()
    ned = NEDMetric()
    cos = CosineMetric()
    
    print("Cosine Similarity:", cos.compute(vec_np, vec_torch))
    print("Normalized Euclidean Distance:", ned.compute(vec_np, vec_torch))

    # x = [20,30,2]
    # y = [1.0,2.0,3.0]
    # print("Dot Product: ", dot_product(x, y))
    # print(euclidean_distance(x, y))
    # # print(NED(x, y))
    # print("Done")