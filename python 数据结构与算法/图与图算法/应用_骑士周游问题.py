from 图结构 import *

### 骑士周游问题

## 构造网格图
def _knightgraph(length):
    graph = Graph()
    for row in range(1, length+1):
        for col in range(1, length+1):
            graph.add(_pos2id(row, col, length))
    for fromkey in graph:
        for tokey in _legalnext(fromkey, length):
            graph.addedge(fromkey, tokey)
    return graph

def _pos2id(row, col, length):
    return (row-1) * length + col

def _id2pos(key, length):
    if key % length == 0:
        return key//length, length
    else:
        return key//length+1, key%length

def _legalnext(key, length):
    row, col = _id2pos(key, length)
    for i in [2, -2]:
        for j in [1, -1]:
            if row+i>0  and row+i<length+1 and col+j>0 and col+j<length+1:
                yield _pos2id(row+i, col+j, length)
    for i in [1, -1]:
        for j in [2, -2]:
            if row+i>0  and row+i<length+1 and col+j>0 and col+j<length+1:
                yield _pos2id(row+i, col+j, length)    

## 寻找周游路径（深度优先遍历）
def _buildpath(graph, cpos, n=0, path=[]):
    graph[cpos]._flag = 1
    path.append(cpos)
    print(path)
    if n < graph.size():
        done = False
        for npos in graph[cpos].getconnections():
            if graph[npos]._flag == 0:
                done = _buildpath(graph, npos, n+1, path)
            if done:
                break
        if not done:
            path.pop()
            print(path)
            graph[cpos]._flag = 0
    else:
        done = True
    return done

## 解决骑士周游问题
def knighttour(length, cpos):
    graph = _knightgraph(length)
    return _buildpath(graph, cpos)

knighttour(3, 1)
