# %%
## 前序遍历（递归）
def preorder(root):
    if root != None:
        print(root.rootval)
        preorder(root.leftchild)
        preorder(root.rightchild)

## 中序遍历（递归）
def inorder(root):
    if root != None:
        postorder(root.leftchild)
        print(root.rootval)
        postorder(root.rightchild)

## 后序遍历（递归）
def postorder(root):
    if root != None:
        postorder(root.leftchild)
        postorder(root.rightchild)
        print(root.rootval)

## 中序遍历输出解析树表达式
def postprint(root):
    if root.leftchild == None:
        return str(root.rootval)
    else:
        return "(" + postprint(root.leftchild) + root.rootval\
            + postprint(root.rightchild) + ")"
postprint(example)

# %%
