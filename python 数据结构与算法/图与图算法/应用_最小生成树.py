graph = [[0, 1, 3, 7, 9],
         [1, 0, 5, 4, 2],
         [3, 5, 0, 7, 1],
         [7, 4, 7, 0, 5],
         [9, 2, 1, 5, 0]]

# %%
## Prim算法
def _findmin(alist):
    minind = 0
    for i in range(len(alist)):
        if alist[i]==0:
            pass
        elif alist[minind]>alist[i] or alist[minind]==0:
            minind = i
    return minind

def prim(graph):
    num = len(graph[0])
    edge, cost = [0]*num, graph[0][:]
    for i in range(1, num):
        # 寻找最短边
        minind = _findmin(cost)
        print("(%s, %s) weight: %s"%(minind, edge[minind], cost[minind]))
        # 更新最短边集
        cost[minind] = 0
        for j in range(len(cost)):
            if cost[j] > graph[minind][j]:
                cost[j] = graph[minind][j]
                edge[j] = minind

prim(graph)

# %%
## Kruskal算法
def _findmatmin(mat):
    q, t = 0, 0
    for i in range(len(mat[0])):
        j = _findmin(mat[i])
        if mat[i][j] == 0:
            pass
        elif mat[q][t]>mat[i][j] or mat[q][t]==0:
            q, t = i, j
    return q, t

def root(i, parent):
    while parent[i] != i:
        i = parent[i]
    return i

import copy
def kruskal(graph):
    mat = copy.deepcopy(graph)
    num = len(mat[0])
    parent = list(range(num))
    while True:
        i, j = _findmatmin(mat)
        if root(i,parent) != root(j,parent):
            print("(%s, %s) weight: %s"%(i, j, mat[i][j]))
            parent[i] = j
            num += -1
            if num == 1:
                break
        mat[i][j] = 0

kruskal(graph)

