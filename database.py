from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # Import cosine similarity

class AbstractDatabase(ABC):
    @abstractmethod
    def insert(self, face_vector, name, summary):
        pass

    @abstractmethod
    def retrieve(self, face_vector):
        pass

    @abstractmethod
    def update(self, face_vector, name, summary):
        pass

    @abstractmethod
    def delete(self, face_vector):
        pass

class InMemoryDatabase(AbstractDatabase):
    def __init__(self):
        self.db = {}

    def insert(self, face_vector, name, summary):
        key = tuple(np.round(face_vector, 4))  # Convert to tuple for dict key
        self.db[key] = {'name': name, 'summary': summary}

    def retrieve(self, face_vector):
        key = tuple(np.round(face_vector, 4))
        return self.db.get(key, None)

    def update(self, face_vector, name, summary):
        key = tuple(np.round(face_vector, 4))
        if key in self.db:
            self.db[key]['summary'] += '\n' + summary
        else:
            self.insert(face_vector, name, summary)

    def delete(self, face_vector):
        key = tuple(np.round(face_vector, 4))
        if key in self.db:
            del self.db[key]
        
    def extract_data_from_face_embedding(self, embedding, threshold=0.6):
            if not self.db:
                return None, None

            embedding = np.array(embedding).reshape(1, -1)[0]  # make it a 1D vector for the custom function
            keys = [np.array(k) for k in self.db.keys()]

            # Compute cosine similarities for each stored key
            similarities = np.array([cosine_similarity(key, embedding) for key in keys])

            best_index = np.argmax(similarities)
            best_score = similarities[best_index]

            if best_score >= threshold:
                best_match_key = tuple(np.round(keys[best_index], 4))
                return self.retrieve(best_match_key), best_match_key

            return None, embedding


def cosine_similarity(embedding1, embedding2):

    """
    Computes the cosine similarity between two face embeddings.
    
    Args:
        embedding1 (np.array): First face embedding
        embedding2 (np.array): Second face embedding. 

    Returns:
        float: Cosine similarity between the two embeddings.
    """

    dot_product = np.dot(embedding1, embedding2.T)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)
