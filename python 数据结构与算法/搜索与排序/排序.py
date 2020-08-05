# %%
## 冒泡排序
def bubblesort(List):
    for i in range(1, len(List)):
        for ind in range(len(List)-i):
            if List[ind] > List[ind+1]:
                List[ind], List[ind+1] = List[ind+1], List[ind]
    return List

## 早停版
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

l = [10,2,9,14,5,199,2,3,1,22,111,2,99]
shortbubble(l[:], ascending=False)

## 快速排序
def partition(List, start, end):
    i, j = start, end
    while i < j:
        while i<j and List[i]<=List[j]:
            j += -1
        if i<j:
            List[i], List[j] = List[j], List[i]
            i += 1
        while i<j and List[i]<=List[j]:
            i += 1
        if i<j:
            List[i], List[j] = List[j], List[i]
            j += -1
    return i
def quicksort(List, start=0, end='default'):
    if end == 'default':
        end = len(List)-1
    if end > start:
        pivot = partition(List, start, end)
        quicksort(List, start, pivot-1)
        quicksort(List, pivot+1, end)
    return List
quicksort(l[:])

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

## 堆排序
def sift(List, k, m):
    i, j = k, 2*k
    while j<=m:
        if j<m and List[j]<List[j+1]:
            j += 1
        if List[i] >= List[j]:
            break
        else:
            List[i], List[j] = List[j], List[i]
            i, j = j, 2*j
def heapsort(List):
    for i in range(len(List)//2-1, -1, -1):
        sift(List, i, len(List)-1)
    for i in range(len(List)-1):
        List[0], List[-i-1] = List[-i-1], List[0]
        sift(List, 0, len(List)-i-2)
    return List
heapsort(l[:])

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
## 归并排序(递归)
def recmergesort(List):
    if len(List) > 1:
        median = len(List)//2
        left = recmergesort(List[:median])
        right = recmergesort(List[median:])
        List = []
        while len(left)*len(right)!=0:
            if left[0] < right[0]:
                List.append(left.pop(0))
            else:
                List.append(right.pop(0))
        return List+left+right
    else:
        return List
recmergesort(l[:])

## 归并排序（非递归）
def merge(List, pivot):
    if len(List) > 1:
        left, right = List[:pivot], List[pivot:]
        List = []
        while len(left)*len(right)!=0:
            if left[0] < right[0]:
                List.append(left.pop(0))
            else:
                List.append(right.pop(0))
        return List+left+right
    else:
        return List
def mergesort(List):
    h = 1
    while True:
        i, count = 0, 0
        while i+2*h <= len(List):
            List[i:i+2*h] = merge(List[i:i+2*h], h)
            i = i+2*h
            count += 1
        if i+h < len(List):
            List[i:] = merge(List[i:], h)
            count += 1
        if count == 1:
            return List
        else:
            h = h*2
mergesort(l[:])

# %%
