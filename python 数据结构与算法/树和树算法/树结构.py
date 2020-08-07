# %%
## 嵌套列表
def ListBinaryTree(r):
    return [r, [], []]
def insertleft(root, branch):
    origin = root.pop(1)
    if len(origin) == 0:
        root.insert(1, [branch,[],[]])
    else:
        root.insert(1, [branch,origin,[]])
    return root
def insertright(root, branch):
    origin = root.pop(2)
    if len(origin) == 0:
        root.insert(2, [branch,[],[]])
    else:
        root.insert(2, [branch,[],origin])
    return root
def getrootval(root):
    return root[0]
def setrootval(root, val):
    root[0] = val
def getleftchild(root):
    return root[1]
def getrightchild(root):
    return root[2]

if __name__ == '__main__':
    r = ListBinaryTree(3)
    insertleft(r,4)
    insertright(r,5)
    insertleft(getleftchild(r),6)
    insertright(getleftchild(r),7)
    insertleft(getrightchild(r),8)
    insertright(getrightchild(r),9)


# %%
## 节点引用
class BinaryTree():
    def __init__(self, val):
        self.rootval = val
        self.leftchild = None
        self.rightchild = None
    def insertleft(self, data):
        newnode = BinaryTree(data)
        newnode.leftchild = self.leftchild
        self.leftchild = newnode
    def insertright(self, data):
        newnode = BinaryTree(data)
        newnode.rightchild = self.rightchild
        self.rightchild = newnode

if __name__ == '__main__':
    r = BinaryTree(3)
    r.insertleft(4)
    r.insertright(5)
    r.leftchild.insertleft((6))
    r.leftchild.insertright((7))
    r.rightchild.insertleft((8))
    r.rightchild.insertright((9))

# %%

