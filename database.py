from abc import ABC, abstractmethod
import numpy as np
import preprocessing
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
        self.db[face_vector] = {'name': name, 'summary': summary}

    def retrieve(self, face_vector):
        return self.db.get(face_vector, None)

    def update(self, face_vector, name, summary):
        if face_vector in self.db:
            self.db[face_vector] = {'name': name, 'summary': summary}

    def delete(self, face_vector):
        if face_vector in self.db:
            del self.db[face_vector]


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

def extract_data_from_face_embedding(database, embedding):
    keys = np.array(database.keys)
    similarities = np.zeros(len(keys))
    best_match = None
    best_score = 0
    for index, k in enumerate(keys):
        similarities[index] = cosine_similarity(k, embedding)
        if similarities[index] > 0.6 and similarities[index] > best_score:
            best_score = similarities[index]
            best_match = k
    json_data = None
    if best_match is not None:
        json_data = database.retrieve(best_match)
    return json_data

