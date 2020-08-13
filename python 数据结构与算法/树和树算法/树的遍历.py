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
## 前序遍历（非递归）
def preorder(root):
    root_stack = Stack()
    while not root_stack.isempty() or root != None:
        while root != None:
            print(root.rootval)
            root_stack.push(root)
            root = root.leftchild
        if not root_stack.isempty():
            root = root_stack.pop().rightchild

## 中序遍历（非递归）
def inorder(root):
    root_stack = Stack()
    while not root_stack.isempty() or root != None:
        while root != None:
            root_stack.push(root)
            root = root.leftchild
        if not root_stack.isempty():
            root = root_stack.pop()
            print(root.rootval)
            root = root.rightchild

## 后序遍历（非递归）
def postorder(root):
    root_stack = Stack()
    while not root_stack.isempty() or root != None:
        while root != None:
            root.flag = 0
            root_stack.push(root)
            root = root.leftchild
        if not root_stack.isempty():
            if root_stack.peek().flag == 0:
                root_stack.peek().flag = 1
                root = root_stack.peek().rightchild
            else:
                print(root_stack.pop().rootval)

# %%
preorder(r)

inorder(r)

postorder(r)
