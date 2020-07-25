## 节点类
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None
    def __str__(self):
        return str(self.data)

## 无序链表
class UnorderedList:
    def __init__(self):
        self.head = None
    def __str__(self):
        node = self.head
        result = ""
        while node!=None:
            result = result + "-->" + str(node.data)
            node = node.next
        return result
    def isempty(self):
        return self.head == None
    def add(self, data):
        oldnode = self.head
        self.head = Node(data)
        self.head.next = oldnode
    def size(self):
        node = self.head
        num = 0
        while node != None:
            num += 1
            node = node.next
        return num
    def search(self, data):
        node = self.head
        while node != None:
            if node.data == data:
                return True
            node = node.next
        return False
    def remove(self, data):
        current = self.head
        previous = self.head
        while current != None:
            if current.data == data:
                if previous == None:
                    self.head = current.next
                else:
                    previous.next = current.next
                current.next = None
                break
            previous = current
            current = current.next
    def append(self, data):
        node = self.head
        if node == None:
            self.head = Node(data)
        else:
            while node.next != None:
                node = node.next
            node.next = Node(data)
    def insert(self, loc, data):
        if loc > self.size():
            raise IndexError("out of index")
        current = self.head
        previous = None
        for i in range(loc):
            previous = current
            current = current.next
        node = Node(data)
        if previous == None:
            self.head = node
        else:
            previous.next = node
        node.next = current
    def index(self, data):
        loc = 0
        node = self.head
        while node != None:
            if node.data == data:
                return loc
            node = node.next
            loc += 1
        return False
    def pop(self, loc=-1):
        if self.isempty():
            raise ValueError("nothing to pop")
        elif loc >= self.size():
            raise IndexError("out of index")            
        current = self.head
        previous = None
        if loc == -1:
            loc = self.size() - 1
        for i in range(loc):
            previous = current
            current = current.next
        if previous == None:
            self.head = current.next
        else:
            previous.next = current.next
        current.next = None
        return current



## 升序链表
class OrderedList:
    def __init__(self):
        self.head = None
    def __str__(self):
        node = self.head
        result = ""
        while node!=None:
            result = result + "-->" + str(node.data)
            node = node.next
        return result
    def isempty(self):
        return self.head == None
    def size(self):
        node = self.head
        num = 0
        while node != None:
            num += 1
            node = node.next
        return num
    def remove(self, data):
        current = self.head
        previous = self.head
        while current != None:
            if current.data == data:
                if previous == None:
                    self.head = current.next
                else:
                    previous.next = current.next
                current.next = None
                break
            elif current.data > data:
                break
            previous = current
            current = current.next
        if current == None:
            raise ValueError("value doesn't exist")
    def search(self, data):
        node = self.head
        while node != None:
            if node.data == data:
                return True
            elif node.data > data:
                break
            node = node.next
        return False
    def add(self, data):
        if self.head == None:
            self.head = Node(data)
        else:
            current = self.head
            previous = None
            while current != None:
                if current.data >= data:
                    if previous == None: 
                        self.head = Node(data)
                        self.head.next = current
                    elif previous.data < data:
                        previous.next = Node(data)
                        previous.next.next = current
                    break
                previous = current
                current = current.next
            if current == None:
                previous.next = Node(data)
    def index(self, data):
        loc = 0
        node = self.head
        while node != None:
            if node.data == data:
                return loc
            elif node.data > data:
                break
            node = node.next
            loc += 1
        raise ValueError("value doesn't exist")
    def pop(self, loc=-1):
        if self.isempty():
            raise ValueError("nothing to pop")
        elif loc >= self.size():
            raise IndexError("out of index")            
        current = self.head
        previous = None
        if loc == -1:
            loc = self.size() - 1
        for i in range(loc):
            previous = current
            current = current.next
        if previous == None:
            self.head = current.next
        else:
            previous.next = current.next
        current.next = None
        return current
    
alist = OrderedList()
alist.add(177)
alist.add(201)
alist.add(199)
alist.add(200)
print(alist)
alist.search(201)
print(alist)