import numpy as np

class Embeddings:

    def __init__(self):
        """
        Initializes the embeddings database.
        """
        self.Vectors = []
        self.Labels = []

    def Add(self, vector, label):
        """
        Adds embedding to embeddings database.
        Args:
            vector: Vector
            label: Label
        """
        self.Vectors.append(vector)
        self.Labels.append(label)

    def Remove(self, label):
        """
        Removes embedding from embeddings database.
        Args:
            label: Label
        """
        index = self.Labels.index(label)
        _ = self.Vectors.pop(index)
        _ = self.Labels.pop(index)

    def Clear(self):
        """
        Clears embeddings database.
        """
        self.Vectors.clear()
        self.Labels.clear()

    def Count(self):
        """
        Returns embeddings database count.
        Returns:
            Count
        """
        return len(self.Vectors)

    def FromDistance(self, vector):
        """
        Score vector from database by Euclidean distance.
        Args:
            vector: Vector

        Returns:
            Label
        """
        length = self.Count()
        minimum = 2147483647
        index = -1

        for i in range(length):
            v = self.Vectors[i]
            d = np.linalg.norm(v - vector)

            if (d < minimum):
                index = i
                minimum = d
        
        label = self.Labels[index] if (index != -1 and self.Labels != []) else ''
        return label, minimum
        
    def FromSimilarity(self, vector):
        """
        Score vector from database by cosine similarity.
        Args:
            vector: Vector

        Returns:
            Label
        """
        length = self.Count()
        maximum = -2147483648
        index = -1

        for i in range(length):
            v = self.Vectors[i]
            a = np.linalg.norm(v)
            b = np.linalg.norm(vector)
            s = np.dot(v, vector) / (a * b)

            if (s > maximum):
                index = i
                maximum = s
        
        label = self.Labels[index] if (index != -1 and self.Labels != []) else ''
        return label, maximum