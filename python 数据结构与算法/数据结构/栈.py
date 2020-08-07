class Stack:
    def __init__(self):
        self.items = []
    def __str__(self):
        return str(self.items)
    def push(self, item):
        self.items.append(item)
    def pop(self):
        return self.items.pop()
    def peek(self):
        return self.items[-1]
    def isempty(self):
        return len(self.items) == 0
    def size(self):
        return len(self.items)
if __name__ == "__main__":
    stack = Stack()
    stack.push(1)
    stack.push('pig')
    print(stack)
    stack.peek()
    stack.isempty()
    stack.size()
    stack.pop()
