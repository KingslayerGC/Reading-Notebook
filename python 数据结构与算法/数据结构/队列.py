## 队列类
class Queue:
    def __init__(self, *item):
        self.items = []
        if item != None:
            for i in item:
                self.enqueue(i)
    def enqueue(self, item):
        self.items.insert(0, item)
    def dequeue(self):
        return self.items.pop()
    def isempty(self):
        return self.items == []
    def size(self):
        return len(self.items)

## 抽杀律游戏
survivor = Queue('笨蛋','傻瓜','铁憨憨','垃圾','臭猪')
def lastsurvivor(survivor, num=7):
    i = 1; print(survivor.items)
    while survivor.size() != 1 :
        if i%num != 0:
            survivor.enqueue(survivor.dequeue())
        else:
            survivor.dequeue()
        #print(survivor.items)
        i+=1
    return survivor.items[0]

print(lastsurvivor(Queue('笨蛋','傻瓜','铁憨憨','垃圾','臭猪')))

## 打印模拟
import random
# 打印机类
class Printer:
    def __init__(self, print_speed):
        self.speed = print_speed
        self.cutask = None
        self.timeremain = 0
    def tick(self):
        if self.timeremain != 0:
            self.timeremain = self.timeremain-1
        if self.timeremain == 0:
            self.cutask = None
    def isempty(self):
        return self.cutask == None
    def timecount(self):
        if self.cutask.quality == 1:
            self.timeremain = self.cutask.pages / self.speed * 2
        else:
            self.timeremain = self.cutask.pages / self.speed
# 打印任务类
class Task:
    def __init__(self, demand_quality):
        self.quality = demand_quality
        self.pages = random.randrange(1,21)
        self.timeremain = 0

def prob(n):
    return random.randrange(n) == 0

def simulation(time, print_speed):
    waittime = []
    tasks = Queue()
    printer = Printer(print_speed)
    for second in range(time):
        printer.tick()
        # 180分之1的概率生成新任务
        if prob(180):
            # 2分之1的概率为高品质任务
            task = Task(prob(2))
            tasks.enqueue(task)
        # 如果打印机空了且打印任务非空，开始打印队首任务
        if printer.isempty() and not tasks.isempty():
            task = tasks.dequeue()
            printer.cutask = task
            printer.timecount()
            waittime.append(printer.timeremain + task.timeremain)
        # 增加等待队列中每一任务的等待时间
        for task in tasks.items:
            task.timeremain = task.timeremain+1
    if len(waittime) == 0:
        print("No Costumers")
    else:
        print("Average Waiting Time：", str(sum(waittime)/len(waittime)))
    return printer, tasks

printer, tasks = simulation(3600, 1/6)

# %%
## 双端队列类
class Deque:
    def __init__(self):
        self.items = []
    def addfront(self, item):
        self.items.append(item)
    def addrear(self, item):
        self.items.insert(0, item)
    def removefront(self):
        return self.items.pop()
    def removerear(self):
        return self.items.pop(0)
    def isempty(self):
        return len(self.items) == 0
    def size(self):
        return len(self.items)

## 检查回文词
def palcheck(string):
    spell = Deque()
    for i in string:
        spell.addrear(i)
    while spell.size() > 1:
        if not spell.removefront() == spell.removerear():
            return False
    return True

palcheck("tangwenxuan")
palcheck("woelleow")
palcheck("abaabalabaaba")
