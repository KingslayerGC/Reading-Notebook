## 节点类
class TreeNode:
    def __init__(self,key,val,left=None,right=None,parent=None):
        self.key = key
        self.value = val
        self.leftchild = left
        self.rightchild = right
        self.parent = parent
    def isleftchild(self):
        return self.parent!=None and self.parent.leftchild==self
    def isrightchild(self):
        return self.parent!=None and self.parent.rightchild==self
    def isroot(self):
        return self.parent == None
    def isleaf(self):
        return self.rightchild==None and self.leftchild==None
    def hastwochildren(self):
        return self.rightchild!=None and self.leftchild!=None
    def __iter__(self):
        if self != None:
            if self.leftchild != None:
                for key in self.leftchild:
                    yield key
            yield self.key
            if self.rightchild != None:
                for key in self.rightchild:
                    yield key


## 二叉排序树类
class BinarySearchTree:
    def __init__(self):
        self.root = None
        self.size = 0
    def length(self):
        return self.size
    def __len__(self):
        return self.size
    def __iter__(self):
        return self.root.__iter__()
    def put(self, key, val, current='default'):
        if current == 'default':
            current = self.root
        if current == None:
            self.root = TreeNode(key, val)
            self.size += 1
        elif key < current.key:
            if current.leftchild == None:
                current.leftchild = TreeNode(key, val)
                current.leftchild.parent = current
                self.size += 1
            else:
                self.put(key, val, current.leftchild)
        elif key > current.key:
            if current.rightchild == None:
                current.rightchild = TreeNode(key, val)
                current.rightchild.parent = current
                self.size += 1
            else:
                self.put(key, val, current.rightchild)
        else:
            current.value = val
    def __setitem__(self, key, val):
        self.put(key, val)
    def get(self, key, current='default', result='value'):
        if current == 'default':
            current = self.root
        if current != None:
            if key < current.key:
                return self.get(key, current.leftchild, result)
            elif key > current.key:
                return self.get(key, current.rightchild, result)
            else:
                if result == 'value':
                    return current.value
                elif result == 'node':
                    return current
                else:
                    raise ValueError("result param should be value or node")
        raise KeyError("key not found")
    def __getitem__(self, key):
        return self.get(key)
    def __contains__(self, key):
        try:
            self.get(key)
            return True
        except KeyError:
            return False
    def delete(self, key, current='default'):
        current = self.get(key, result='node')
        # 如果为叶子节点
        if current.isleaf():
            if current.isleftchild():
                current.parent.leftchild = None
            else:
                current.parent.rightchild = None
            del current
            self.size += -1
        # 如果只有左子树
        elif current.rightchild == None:
            if current.isroot():
                self.root = current.leftchild    
                current.leftchild.parent = None
            elif current.isleftchild():
                current.parent.leftchild = current.leftchild
                current.leftchild.parent = current.parent
            else:
                current.parent.rightchild = current.leftchild
                current.leftchild.parent = current.parent
            del current
            self.size += -1
        # 如果只有右子树
        elif current.leftchild == None:
            if current.isroot():
                self.root = current.rightchild
                current.rightchild.parent = None
            elif current.isleftchild():
                current.parent.leftchild = current.rightchild
                current.rightchild.parent = current.parent
            else:
                current.parent.rightchild = current.rightchild
                current.rightchild.parent = current.parent
            del current
            self.size += -1 
        # 如果左右子树均非空
        else:
            replace = current.rightchild
            while replace.leftchild != None:
                replace = replace.leftchild
            replacekey, replaceval = replace.key, replace.value
            self.delete(replace.key)
            current.key, current.value = replacekey, replaceval

                
            

mytree = BinarySearchTree()

mytree[4]="blue"

mytree[3]="red"

mytree[8]="yellow"

mytree[6]='comeon'

mytree[7]='why'

mytree[10]='raise'

for key in mytree:
    print(key)
for key in mytree.get(8, result='node'):
    print(key)
