# %%
## 顺序搜索
def search(List, item):
    for i in range(len(List)):
        if List[i] == item:
            return i
    return False

def ordersearch(List, item):
    for i in range(len(List)):
        if List[i] == item:
            return i
        elif List[i] > item:
            break
    return False

# %%
## 二分法搜索
def cutsearch(List, item):
    if len(List) == 1:
        return List[0] == item
    elif List[len(List)//2] > item:
        return cutsearch(List[:len(List)//2], item)
    elif List[len(List)//2] < item:
        return cutsearch(List[len(List)//2+1:], item)
    else:
        return True   

cutsearch([1,2,3,4,5,6,6,6,7,8], 8)
cutsearch([1,2,3,4,5,6,6,6,7,8], 100)

# %%
## 键-值对搜索（散列）
class Hashtable:
    def __init__(self):
        self.size = 3
        self.slots = [None] * self.size
        self.data = [None] * self.size
    def __setitem__(self, key, data):
        current_slot = self.hashfunction(key)
        recycle_num = 0
        while self.slots[current_slot]!=None and self.slots[current_slot]!=key:
            current_slot = self.rehash(current_slot)
            recycle_num += 1
            if recycle_num > self.size:
                raise ValueError("the dict has been fully filled")
        self.slots[current_slot] = key
        self.data[current_slot] = data
    def __getitem__(self, key):
        current_slot = self.hashfunction(key)
        while self.slots[current_slot] != None:
            if self.slots[current_slot] == key:
                return self.data[current_slot]
            current_slot = self.rehash(current_slot)        
        raise KeyError("key not found")
    def __delitem__(self, key):
        current_slot = self.hashfunction(key)
        while self.slots[current_slot] != None:
            if self.slots[current_slot] == key:
                self.slots[current_slot] = None
                self.data[current_slot] = None    
                return
            current_slot = self.rehash(current_slot)
        raise KeyError("key not found") 
    def hashfunction(self, key):
        return key % self.size
    def rehash(self, oldhash):
        return (oldhash + 1) % self.size
t = Hashtable()
t[32] = 111
t[21] = 20
t[86] = 8
t[32] = 11
t[21]
del t[21]
