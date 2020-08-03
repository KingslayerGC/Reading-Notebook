## 节点类
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
    def __str__(self):
        return str(self.data)


## 链队列
class ChainQueue:
    def __init__(self):
        self.head = None
        self.rear = None
    def __str__(self):
        node = self.head
        output = ['队首-->']
        while node != None:
            output.append(node.data)
            node = node.next
        output.append('-->队尾')
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
    def enqueue(self, data):
        node = self.rear
        if node == None:
            self.rear = Node(data)
            self.head = self.rear
        else:
            node.next = Node(data)
            self.rear = node.next
    def dequeue(self):
        node = self.head
        if node == None:
            raise RuntimeError("nothing to dequeue")
        if node == self.rear:
            self.rear = None
        self.head = node.next
        node.next = None
        return node

queue = ChainQueue()
queue.enqueue(1)
queue.dequeue()
queue.enqueue('dasda')
queue.enqueue(1231)
print(queue)



