# %%
## 冒泡排序
def bubblesort(List):
    for i in range(1, len(List)):
        for ind in range(len(List)-i):
            if List[ind] > List[ind+1]:
                List[ind], List[ind+1] = List[ind+1], List[ind]
    return List
# 早停版
def shortbubble(alist, ascending=True):
    List = alist[:]
    for i in range(1, len(List)):
        exchange = False
        for ind in range(len(List)-i):
            if List[ind] > List[ind+1]:
                exchange = True
                List[ind], List[ind+1] = List[ind+1], List[ind]
        if not exchange:
            break
    if not ascending:
        return List[::-1]
    else:
        return List

l = [10,2,9,4,5,199,2,3,1,5,6,4,3,2,1,23,111]
shortbubble(l[:], ascending=False)

# %%
## 选择排序
def selectsort(List, ascending=True):
    for i in range(len(List)-1):
        maxind = 0
        for ind in range(1, len(List)-i):
            if List[ind] > List[maxind]:
                maxind = ind
        List[maxind], List[len(List)-i-1] = List[len(List)-i-1], List[maxind]
    if not ascending:
        return List[::-1]
    else:
        return List
selectsort(l[:], ascending=False)

# %%
## 插入排序
def insertsort(List, ascending=True):
    for i in range(1, len(List)):
        current = List[i]
        for ind in range(i-1, -1, -1):
            if current < List[ind]:
                List[ind+1] = List[ind]
            else:
                ind += 1
                break
        List[ind] = current
    if not ascending:
        return List[::-1]
    else:
        return List
insertsort(l[:], ascending=False)

# %%
## 希尔排序(子序列+插入排序)
def shellsort(List, ascending=True):
    subcount = len(List)//2
    while subcount > 0:
        for i in range(subcount):
            List[i::subcount] = insertsort(List[i::subcount])
        subcount = subcount // 2
    if not ascending:
        return List[::-1]
    else:
        return List
shellsort(l[:], ascending=False)
    
# %%
## 归并排序
def mergesort(List):
    if len(List) > 1:
        median = len(List)//2
        left = mergesort(List[:median])
        right = mergesort(List[median:])
        List = []
        while len(left)*len(right)!=0:
            if left[0] < right[0]:
                List.append(left.pop(0))
            else:
                List.append(right.pop(0))
        return List+left+right
    else:
        return List
mergesort(l[:])

# %%
## 快速排序
