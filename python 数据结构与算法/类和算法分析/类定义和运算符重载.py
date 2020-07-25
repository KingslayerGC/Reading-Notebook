## 分数类
class fraction:
    def __init__(self, top, bottom):
        if bottom == 0:
            raise ValueError("Mathematics mistake")
        if type(top)!=int or type(bottom)!=int:
            raise ValueError("Please input integer")
        self.num = top
        self.den = bottom
    # 打印
    def __str__(self):
        m, n = self.gcd()
        return str(int(m))+"/"+str(int(n))
    # 化简
    def gcd(self):
        if self.num == 0:
            return self.num, self.den
        if self.num > self.den:
            m = self.num
            n = self.den
        else:
            m = self.den
            n = self.num
        while(m%n!=0):
            k = m%n
            m = n
            n = k
        return self.num/n, self.den/n
    def __add__(self, f2):
        return fraction(self.num*f2.den + self.den*f2.num, self.den*f2.den)
    def __sub__(self, f2):
        return fraction(self.num*f2.den - self.den*f2.num, self.den*f2.den)
    def __mul__(self, f2):
        return fraction(self.num*f2.num, self.den*f2.den)
    def __truediv__(self, f2):
        return fraction(self.num*f2.den, self.den*f2.num)
    
a = fraction(1,2)
b = fraction(1,3)

print(a/b)
