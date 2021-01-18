import numpy as np

class Embeddings:
    def __init__(self):
        """[summary]
        """
        self.Vectors = []
        self.Labels = []

    def Add(self, vector, label):
        """[summary]

        Args:
            vector ([type]): [description]
            label ([type]): [description]
        """
        self.Vectors.append(vector)
        self.Labels.append(label)

    def Remove(self, label):
        """[summary]

        Args:
            label ([type]): [description]
        """
        index = self.Labels.index(label)
        _ = self.Vectors.pop(index)
        _ = self.Labels.pop(index)

    def Clear(self):
        """[summary]
        """
        self.Vectors.clear()
        self.Labels.clear()

    def Count(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return len(self.Vectors)

    def FromDistance(self, vector):
        """[summary]

        Args:
            vector ([type]): [description]

        Returns:
            [type]: [description]
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
        """[summary]

        Args:
            vector ([type]): [description]

        Returns:
            [type]: [description]
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