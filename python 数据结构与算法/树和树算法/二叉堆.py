### 与堆排序原理相同
## 二叉堆
class BinHeap():
    def __init__(self):
        self.items = []
    def __str__(self):
        return str(self.items)
    def size(self):
        return len(self.items)
    def isempty(self):
        return len(self.items) == 0
    def sift(self, k, m):
        i, j = k, 2*k+1
        while j<=m:
            if j<m and self.items[j]>self.items[j+1]:
                j += 1
            if self.items[i] <= self.items[j]:
                break
            else:
                self.items[i], self.items[j] = self.items[j], self.items[i]
                i, j = j, 2*j+1
    def insert(self, data):
        self.items.append(data)
        i, j = len(self.items)//2-1, len(self.items)-1
        while self.items[i]>self.items[j] and i>=0:
            self.items[i], self.items[j] = self.items[j], self.items[i]
            j, i = i, (i-1)//2
    def getmin(self):
        return self.items[0]
    def popmin(self):
        self.items[0], self.items[-1] = self.items[-1], self.items[0]
        self.sift(0, len(self.items)-2)
        return self.items.pop()
    def buildheap(self, List):
        self.items = List[:]
        for i in range(len(self.items)//2-1, -1, -1):
            self.sift(i, len(self.items)-1)

l = [10,2,9,14,5,199,2,3,1,22,111,2,99]
tree = BinHeap()
for i in l:
    tree.insert(i)
tree.buildheap(l)
print(tree)
