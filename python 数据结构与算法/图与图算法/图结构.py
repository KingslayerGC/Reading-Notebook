## 顶点类
class Vertex():
    def __init__(self, key):
        self.id = key
        self.connection = {}
    def addneighbor(self, key, weight):
        self.connection[key] = weight
    def __str__(self):
        string = ""
        for key, weight in self.connections:
            string += ('to %s with weight %s\n'(key, weight))
    def getconnections(self):
        self.connection.keys()
    def getweight(self, key):
        return self.connections[key]

## 图类
class Graph():
    def __init__(self):
        self.vlist = {}
    def add(self, vertex):
        


