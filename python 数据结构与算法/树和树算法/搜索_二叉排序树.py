## 节点类
class TreeNode:
    def __init__(self,key,val,left=None,right=None,parent=None):
        self.key = key
        self.value = val
        self.leftchild = left
        self.rightchild = right
        self.parent = parent
    def isleftchild(self):
        return self.parent!=None and self.parent.leftChild==self
    def isrightchild(self):
        return self.parent!=None and self.parent.rightChild==self
    def isroot(self):
        return self.parent == None
    def isleaf(self):
        return self.rightChild==None and self.leftChild==None
    def hastwochildren(self):
        return self.rightChild!=None and self.leftChild!=None
    def replaceNodeData(self,key,value,lc,rc):
        self.key = key
        self.value = value
        self.leftchild = lc
        self.righchild = rc
        if self.hasLeftChild():
            self.leftChild.parent = self
        if self.hasRightChild():
            self.rightChild.parent = self


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


mytree = BinarySearchTree()
mytree[3]="red"
mytree[4]="blue"
mytree[6]="yellow"
mytree[2]="at"

print(mytree[6])
print(mytree[2])
