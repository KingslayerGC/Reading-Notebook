# %%
## 解析树
import operator
def parsetree(string):
    father_stack = Stack()
    root = BinaryTree(None)
    current = root
    father_stack.push(current)
    for i in list(string):
        if i  == "(":
            father_stack.push(current)
            current.insertleft(None)
            current = current.leftchild
        elif i == ")":
            current = father_stack.pop()
        elif i in ["+", "-", "*", "/"]:
            current.rootval = i
            father_stack.push(current)
            current.insertright(None)
            current = current.rightchild
        else:
            try:
                current.rootval = int(i)
                current = father_stack.pop()
            except ValueError:
                raise ValueError("cat not parse %s" %i)
    return root
def evaluate(root):
    opers = {'+':operator.add, '-':operator.sub, '*':operator.mul, '/':operator.truediv}
    if root.leftchild == None:
        return root.rootval
    else:
        leftval = evaluate(root.leftchild)
        rightval = evaluate(root.rightchild)
        return opers[root.rootval](leftval, rightval)

example = parsetree("(((3+4)*(5-3))/3)")
evaluate(example)

