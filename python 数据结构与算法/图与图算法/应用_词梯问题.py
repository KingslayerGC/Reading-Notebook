# %%
from 队列 import Queue
from 图结构 import *

## 构造词图
def buildgraph(wordFile):
    d = {}
    g = Graph()
    with open(wordFile,'r') as file_object:
        for word in file_object:
            if word[-1] == "\n":
                word = word[:-1]
            word = word.lower()
            g.add(word)
            for i in range(len(word)):
                mode = word[:i] + '_' + word[i+1:]
                if mode in d:
                    d[mode].append(word)
                else:
                    d[mode] = [word]
    # add vertices and edges for words in the same bucket
    for bucket in d:
        for word1 in d[bucket]:
            for word2 in d[bucket]:
                if word1 != word2:
                    g.addedge(word1,word2)
    return g

graph = buildgraph("word.txt")

# %%
## 寻找最短词梯（广度优先遍历）
def _buildladder(graph, start, end):
    queue = Queue()
    cword = start
    while True:
        for nword in graph[cword].getconnections():
            if nword == end:
                graph[nword]._parent = cword
                return
            if graph[nword]._flag == 0:
                graph[nword]._flag = 1
                graph[nword]._parent = cword
                queue.enqueue(nword)
        if not queue.isempty():
            cword = queue.dequeue()
        else:
            break

def _removeladder(graph):
        for key in graph:
            graph[key]._parent = None
            graph[key]._flag = 0
          
def showladder(graph, start, end):
    _buildladder(graph, start, end)
    if graph[end]._parent != None:
        ladder = [end]
        word = end
        while word != start:
            word = graph[word]._parent
            ladder.append(word)
        string = ""
        for word in ladder[::-1]:
            string += word + "-->"
        _removeladder(graph)
        print("the possible shortest ladder is ", string[:-3])
        return len(ladder)-1
    else:
        _removeladder(graph)
        raise RuntimeError("can't find any possible ladder")

showladder(graph, "pole", "pool")


