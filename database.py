from abc import ABC, abstractmethod

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
