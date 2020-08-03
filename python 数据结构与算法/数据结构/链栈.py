## 节点类
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
    def __str__(self):
        return str(self.data)

## 链栈
class ChainStack:
    def __init__(self):
        self.head = None
    def __str__(self):
        node = self.head
        output = ['栈顶']
        while node != None:
            output.append(node.data)
            node = node.next
        output.append('栈底')
        return str(output)
    def isempty(self):
        return self.head == None
    def size(self):
        count = 0
        node = self.head
        while node != None:
            count += 1
            node = node.next
        return count
    def push(self, data):
        node = self.head
        self.head = Node(data)
        self.head.next = node
    def pop(self):
        node = self.head
        if node == None:
            raise RuntimeError("nothing to pop")
        self.head = node.next
        node.next = None
        return node
    def peek(self):
        if self.isempty():
            raise RuntimeError("nothing in this chainstack")
        return self.head.data

stack = ChainStack()
stack.push('sdasd')
print(stack)
stack.push(11)
stack.push(1919)
print(stack)
stack.peek()
stack.pop()
print(stack)
            
                





