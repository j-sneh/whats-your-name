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
        """
        Searches for the closest face embedding using cosine similarity.

        Args:
            embedding (numpy.array): The face embedding vector to search for.
            threshold (float): Minimum cosine similarity threshold for a match.

        Returns:
            dict or None: The matching entry from the database, or None if no match is found.
        """
        if not self.db:
            return None  # Return None if database is empty

        embedding = np.array(embedding).reshape(1, -1)  # Ensure it's a 2D array
        keys = np.array([np.array(k) for k in self.db.keys()])  # Convert stored keys back to numpy arrays

        # Compute cosine similarities
        similarities = cosine_similarity(keys, embedding).flatten()

        # Find best match
        best_index = np.argmax(similarities)
        best_score = similarities[best_index]

        if best_score >= threshold:
            best_match_key = tuple(keys[best_index])  # Convert back to tuple for retrieval
            return self.retrieve(best_match_key)

        return None  # No match found



def cosine_similarity(embedding1, embedding2):

    """
    Computes the cosine similarity between two face embeddings.
    
    Args:
        embedding1 (np.array): First face embedding
        embedding2 (np.array): Second face embedding. 

    Returns:
        float: Cosine similarity between the two embeddings.
    """

    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)
