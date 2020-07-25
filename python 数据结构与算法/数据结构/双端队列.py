## 双端队列类
class Deque:
    def __init__(self):
        self.items = []
    def addfront(self, item):
        self.items.append(item)
    def addrear(self, item):
        self.items.insert(0, item)
    def removefront(self):
        return self.items.pop()
    def removerear(self):
        return self.items.pop(0)
    def isempty(self):
        return len(self.items) == 0
    def size(self):
        return len(self.items)

## 检查回文词
def palcheck(string):
    spell = Deque()
    for i in string:
        spell.addrear(i)
    while spell.size() > 1:
        if not spell.removefront() == spell.removerear():
            return False
    return True

palcheck("tangwenxuan")
palcheck("woelleow")
palcheck("abaabalabaaba")
