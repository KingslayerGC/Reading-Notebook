# %%
## 递归求和
def list_sum(alist):
    if len(alist) == 0:
        raise ValueError("nothing to sum")
    elif len(alist) == 1:
        return alist[0]
    else:
        return alist[0] + list_sum(alist[1:])

list_sum([0,1,2,3,4])

## 递归转化进制
def to_str(n, base):
    if n < base:
        return str(n)
    else:
        return to_str(n//base, base) + str(n%base)

# %%
## 递归检查回文
# 检查是否字母
def isletter(string):
    if len(string) != 1:
        raise ValueError("input length must be one")
    return (((ord(string)>=65) and (ord(string)<=90))\
            or ((ord(string)>=97) and (ord(string)<=122)))
# 删除非字母符号
def delnonl(string):
    if len(string) == 1:
        if isletter(string):
            return string
        else:
            return ''
    else:
        return delnonl(string[0]) + delnonl(string[1:])
# 检查是否回文词
def palcheck(string):
    if len(string) <= 1:
        return True
    elif abs(ord(string[0])-ord(string[-1])) in [0, 32]:
        return palcheck(string[1:-1])
    else:
        return False
if palcheck(delnonl(input("please input the word/sentense you want to check"))):
     print("this is a pal")
else:
    print("this is not a pal")

# %%
## 海龟绘图
import turtle
myTurtle = turtle.Turtle()
myWin = turtle.Screen()
def drawSpiral(myTurtle, lineLen):
    if lineLen > 0:
        myTurtle.forward(lineLen)
    myTurtle.right(90)
    drawSpiral(myTurtle,lineLen-5)
drawSpiral(myTurtle,100)
myWin.exitonclick(), turtle.done(), turtle.bye()

# %%
## 海龟分叉树
import turtle
def tree(branchLen, Width, t):
    if branchLen > 5:
        t.pensize(Width)
        t.forward(branchLen)
        t.right(25)
        tree(branchLen-10, Width-1, t)
        t.left(50)
        tree(branchLen-10, Width-1, t)
        t.right(25)
        t.backward(branchLen)
def main():
    t = turtle.Turtle()
    #myWin = turtle.Screen()
    t.left(90)
    t.up()
    t.backward(100)
    t.down()
    t.color("green")
    tree(75, 10, t)
    #myWin.exitonclick()
    turtle.done(), turtle.bye()
main()

# %%
## 谢尔宾斯基三角形
import turtle
def draw_triangle(points, color, my_turtle):
    my_turtle.fillcolor(color)
    my_turtle.up()
    my_turtle.goto(points[0][0],points[0][1]) 
    my_turtle.down()
    my_turtle.begin_fill()
    my_turtle.goto(points[1][0],points[1][1])
    my_turtle.goto(points[2][0],points[2][1])
    my_turtle.goto(points[0][0],points[0][1])
    my_turtle.end_fill()
def mid_points(p1,p2):
    return ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
def sierpinski(points, degree, my_turtle):
    color_map = ['blue', 'red', 'green', 'white', 'yellow', 'violet', 'orange']
    draw_triangle(points, color_map[degree], my_turtle)
    if degree > 0:
        sierpinski([points[0],mid_points(points[0],points[1]),
                    mid_points(points[0],points[2])], degree-1, my_turtle)
        sierpinski([points[1],mid_points(points[1],points[0]),
                    mid_points(points[1],points[2])], degree-1, my_turtle)
        sierpinski([points[2],mid_points(points[2],points[1]),
                    mid_points(points[0],points[2])], degree-1, my_turtle)
def main():
    my_turtle = turtle.Turtle()
    #my_win = turtle.Screen()
    my_points = [[-100, -50],[0, 50],[100, -50]]
    sierpinski(my_points, 3, my_turtle)
    #my_win.exitonclick()
    turtle.done(), turtle.bye()
main()

# %%
## 河内塔问题
def moveTower(height, fromPole, toPole, withPole):
    if height >= 1:
        moveTower(height-1, fromPole, withPole, toPole)
        moveDisk(fromPole,toPole)
        moveTower(height-1, withPole, toPole, fromPole)
def moveDisk(fp, tp):
    print("moving disk from", fp, "to", tp)
moveTower(3,"A","B","C")

# %%
## 最少硬币找零问题（非递归）
def makechange(coinvalues, amount):
    if 1 not in coinvalues:
        raise RuntimeError("Coinvalues must contain 1")
    mincoinscount = list(range(amount+1))
    coinscontain = [[]]*(amount+1)
    for i in range(amount+1):
        coinscontain[i] = [1]*i
    for value in range(1, amount+1):
        coins = [coin for coin in coinvalues if coin<=value]
        for coin in coins:
            if mincoinscount[value-coin]+1 < mincoinscount[value]:
                mincoinscount[value] = mincoinscount[value-coin]+1
                coinscontain[value] = coinscontain[value-coin]+[coin]
    return mincoinscount, coinscontain
def printmethod(coinvalues, amount):
    mincoinscount, coinscontain = makechange(coinvalues, amount)
    coinscontain = coinscontain[amount]
    coinscontain = dict(zip(
        set(coinscontain),
        [coinscontain.count(coin) for coin in set(coinscontain)]
        ))
    print("对于%u单位的找零，最少需给出%u枚硬币，可能的方案如下："\
          %(amount, mincoinscount[amount]))
    for k, v in coinscontain.items():
        print("%u单位硬币%u个" %(k, v))
printmethod([1,2,5], 13)

