from 栈 import Stack
from 图结构 import Graph, Vertex
class DirGraph(Graph):
    def addedge(self, fromkey, tokey, weight=None):
        if fromkey in self and tokey in self:
            self[fromkey].addneighbor(tokey, weight)
            self[tokey].income += 1
        else:
            raise KeyError("key not found")

def gengraph():
    graph = DirGraph()
    for i in range(6):
        graph.add(i)
    graph.addedge(1, 0)
    graph.addedge(1, 3)
    graph.addedge(2, 0)
    graph.addedge(2, 3)
    graph.addedge(3, 0)
    graph.addedge(3, 5)
    graph.addedge(4, 2)
    graph.addedge(4, 3)
    graph.addedge(4, 5)
    return graph

origraph = gengraph()

# %%
## AOV网的拓扑排序
import copy
def topsort(origraph):
    graph = copy.deepcopy(origraph)
    count = 0
    stack = Stack()
    for key in graph:
        if graph[key].income == 0:
            stack.push(key)
    string = "a possible topsort is "
    while not stack.isempty():
        ckey = stack.pop()
        count += 1
        string += str(ckey) + "-->"
        for tokey in graph[ckey].getconnections():
            graph[tokey].income += -1
            if graph[tokey].income == 0:
                stack.push(tokey)
    if count < graph.size():
        print("there is a circle path in current graph")
    else:
        print(string[:-3])

topsort(origraph)
