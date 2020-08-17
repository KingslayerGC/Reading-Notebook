inf = float('inf')
graph = [[0, 1, inf, inf, 8],
         [inf, 0, 1, inf, 3],
         [inf, inf, 0, 1, inf],
         [inf, inf, inf, 0, inf],
         [inf, inf, inf, 1, 0]]

# %%
## Dijkstra算法
def _findmin(alist):
    minind = 0
    for i in range(len(alist)):
        if alist[i]==0:
            pass
        elif alist[minind]>alist[i] or alist[minind]==0:
            minind = i
    return minind

def dijkstra(graph, start):
    num = len(graph[0])
    path = [str(start)+str(i) for i in range(num)]
    dist = graph[start][:]
    # 按距离从小到大每次加入一个顶点
    for i in range(num-1):
        minind = _findmin(dist)
        # 若是自身不可达，则不打印
        if dist[minind] != float('inf'):
            print("the shortest path from %s to %s: %s the distance is %s"
                  %(start, minind, path[minind], dist[minind]))
            for j in range(num):
                if dist[j] > dist[minind] + graph[minind][j]:
                    dist[j] = dist[minind] + graph[minind][j]
                    path[j] = path[minind] + str(j)
            dist[minind] = 0

dijkstra(graph, 1)

# %%
## Floyd算法
import copy
def floyd(graph, start=None):
    num = len(graph[0])
    dist, path = copy.deepcopy(graph), copy.deepcopy(graph)
    for i in range(num):
        for j in range(num):
            dist[i][j] = graph[i][j]
            path[i][j] = str(i)+str(j)
    # 按任意顺序每次加入一个顶点
    for n in range(num):
        for i in range(num):
            for j in range(num):
                if dist[i][j] > dist[i][n] + dist[n][j]:
                    dist[i][j] = dist[i][n] + dist[n][j]
                    path[i][j] = path[i][n] + path[n][j][1:]
    for i in range(num):
        if start in [None, i]:
            for j in range(num):
                # 若不可达，则不打印
                if dist[i][j] not in [0, float('inf')]:
                    print("the shortest path from %s to %s: %s the distance is %s"
                          %(i, j, path[i][j], dist[i][j]))

floyd(graph, 1)
