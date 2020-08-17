## 顶点类
class Vertex():
    def __init__(self, key):
        self.id = key
        self.connections = {}
        self._parent = None
        self._flag = 0
    def addneighbor(self, key, weight=None):
        self.connections[key] = weight
    def __str__(self):
        string = "id:%s\n"%self.id
        for key, weight in self.connections.items():
            string += ('to %s with weight %s\n'%(key, weight))
        return string
    def getconnections(self):
        return self.connections.keys()
    def getweight(self, key):
        return self.connections[key]

## 图类(邻接表)
class Graph():
    def __init__(self):
        self.vlist = {}
    def __contains__(self, key):
        return key in self.vlist
    def __iter__(self):
        return self.vlist.__iter__()
    def __str__(self):
        string = ""
        for key in self:
            string += self[key].__str__() + "\n"
        return string[:-1]
    def __getitem__(self, key):
        if key in self:
            return self.vlist[key]
        else:
            raise KeyError("key not found")
    def size(self):
        count = 0
        for key in self:
            count += 1
        return count
    def add(self, key):
        self.vlist[key] = Vertex(key)
    def addedge(self, fromkey, tokey, weight=None):
        if fromkey in self and tokey in self:
            self[fromkey].addneighbor(tokey, weight)
        else:
            raise KeyError("key not found")

if __name__ == '__main__':
    graph = Graph()
    graph.add(1)
    graph.add(2)
    graph.add(3)
    graph.addedge(2, 1, 123)
    print(graph)
    for key in graph:
        print(key)
