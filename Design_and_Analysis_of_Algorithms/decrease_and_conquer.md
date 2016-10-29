# 分治法

## 基本思想

- 将问题的实例划分为几个较小的实例，最好拥有同样的规模
- 对这些较小的实例进行求解(一般采用递归的方法，但问题规模很小的问题，也会采用其他方法)
- 如果有必要的话，合并这些子问题

## 合并排序

对于一个需要排序的数组，将它一分为二，并对每个子数组递归地进行合并排序，然后将排好序的数组进行合并。

### 伪代码

- 排序算法
```
MergeSort(A[0...n-1])
// 递归地调用MergeSort来对数组排序
// 输入：数组A[0...n-1]
// 输出：一个非降序排列的数组
if n>1:
  copy A[0...n/2-1] to B[0...n/2-1]
  copy A[n/2..n-1] to C[0...n/2-1]
  MergeSort(B[0...n/2-1])
  MergeSort(C[0...n/2-1])
  Merge(B,C,A)
```
- 合并算法
```
Merge(B[0...p-1],C[0...q-1],A[0...p+q-1])
// 将两个有序数组合并为一个有序数组
// 输入：两个有序数组B[0...p-1]和C[0...q-1]
// 输出：A[0...p+q-1]中已经有序存放了B和C中的元素
i<-0;j<-0;k<-0
while i<p and i<q do:
  if B[i]=<C[j]:
    A[k]<-B[i];i<-i+1
  else:
    A[k]<-C[j];j<-j+1
  k<-k+1
if i=p:
    copy C[j...q-1] to A[k...p+q-1]
  else:
    copy B[i...p-1] to A[k...p+q-1]
```

## 快速排序

不像合并排序是按照元素在数组中的位置对它们进行划分，快速排序按照元素的值对它们进行划分。前面的分区都小于等于中轴的值，后面的分区都大于等于中轴的值。

### 伪代码

- 排序算法

```
Quicksort(A[l...r])
// 用Quicksort对子数组排序
// 输入：数组A[0...n-1]中的子数组A[l...r]，由左右下标l和r定义
// 输出：非降序排列的子数组A[l...r]
if l<r:
  s<-Partition(A[l...r])  // s是分裂位置
Quicksort(A[l...s-1])
Quicksort(A[s+1...r])
```

- 划分算法

```
Partition(A[l...r])
// 以第一个元素为中轴对数组A[l...r]进行分区
// 输入：子数组A[l...r]，由左右下标l和r定义
// 输出：A[l...r]的一个分区，分裂点的位置作为函数的返回值
p<-A[l]
i<-l;j<-r+1
repeat
  repeat i<-i+1 until A[i]>=p
  repeat j<-j-1 until A[j]<=p
  swap(A[i],A[j])
until i>=j
swap(A[i],A[j]) // 当i>=j撤销最后一次交换
swap(A[l],A[j])
return j
```

## 二分查找

通过比较查找键和中间值元素的值来完成查找工作。如果它们的值相等，则算法结束；否则如果K<A[m]，就对前半部分执行该操作，如果K>A[m]，则对数组后半部分执行该操作。

### 伪代码

- 非递归
```
Binarysearch(A[0...n-1],K):
// 实现非递归的二分查找
// 输入：一个升序数组A[0...n-1]和一个查找键K
// 输出；一个数组元素的下标，该元素等于K；如果没有这样的元素，则返回-1
l<-0,r<-n-1
while l<=r do:
  m<-(l+r)/2
  if K=A[m]
    return m
  else if K<A[m]
    r<-m-1
  else
    l<-m+1
return -1
```
- 递归

```
Binarysearch(A[0...n-1],K):
// 实现递归的二分查找
// 输入：一个升序数组A[l...r]和一个查找键K，由l和r限制
// 输出；一个数组元素的下标，该元素等于K；如果没有这样的元素，则返回-1
m<-(l+r)/2
if K=A[m]:
  return m
else if K<A[m]:
  return Binarysearch(A[l, m-1],K)
else
  return Binarysearch(A[m+1,r],K)
```

## 练习

#### 士兵排队问题

在一个划分成网格的操场上，n个士兵散乱地站在网格点上。网格点由整数坐标(x,y)表示。士兵们可以沿网格边上、下、左、右移动一步，但在同一时刻任一网格点上只能有一名士兵。按照军官的命令，士兵们要整齐地列成一个水平队列，即排列(x,y),(x+1,y),…,(x+n-1,y)。如何选择x 和y的值才能使士兵们以最少的总移动步数排成一列。 编程计算使所有士兵排成一行需要的最少移动步数。

- 解题思路：

对于y轴：

最优步数：S=|Y0-M|+|Y1-M|+|Y2-M|+ …… …… +|Yn-1-M|

我们不难发现当M为y轴坐标的中值的时候，移动的总步数最小。

对于x轴：

由题意可假设每个士兵对应的最终x轴坐标分别为：

x,x+1,x+2,...x+n-1

最优步数：S=(x1-(x))+(x2-(x+1))+...+(xn-(x+n-1))=(x1-x)+((x2-1)-x)+...+((xn-n-1)-x)，这样就将问题转化为和y轴相同的问题，因此只要求得x1,x2-1,x3-2,...,xn-(n-1)的中位数即可(这里的x1,x2-1,...,xn-(n-1)就相当于y1,y2,...,yn-1)，最后根据中位数即可得到最小的移动的步数。

- 伪代码：

```
Soldiersqueue(x[0...n-1],y[0...n-1]):
// 求出士兵移动的最小步数
// 输入：每个士兵的x坐标与y坐标
// 输出：最小的移动步数之和
Quicksort(y[0...n-1])
y_middle=y[(n-1)/2]   // 求出y轴坐标上的中值
Quicksort(x[0...n-1]) // 对x轴坐标进行排序
for i from 0 to n-1:
  Xp[i]<-x[i]-i  // 将移动步数保存到数组Xp中
Quicksort(Xp[0...n-1])
x_middle=Xp[(n-1)/2]   // 求Xp数组的中位数
for i from 0 to n-1
  shortestPath+=|Xp[i]-x_middle|+|y[i]-y_middle|  // 求最终的最短步数
```
**NOTE:**在这个问题中，求解中位数和排序用到了分治的思想，但问题整体并不是分治的思想。

### 分治法求众数

在一个元素组成的表中，出现次数最多的元素称为众数。试写一个寻找众数的算法，并分析其计算复杂性。

- 解题思路

选取一个元素作为中轴，然后以该元素为中心划分该数组，所有小于等于该中轴的元素都被划分到中轴的左边，所有大于等于中轴的元素被划分到中轴的右边，另外等于该中轴的元素都聚集在中轴位置。如果左边的元素大于中轴的个数，则继续对左边重复上述操作；如果右边元素的个数大于中轴的个数，则继续对右边重复上述的操作。因为如果左边(右边)的元素个数小于中轴的个数，则左边(右边)不可能存在某个元素的个数大于中轴的个数。只要递归地进行上述操作，就可以找到众数。

- 伪代码

递归函数：

```
GetMode(A[0..n-1], maxCount(out), finalMode(out)) // out代表输出参数
// 递归地求一个数组中的众数
// 输入：数组A[0...n-1]
// 输出：众数及其出现次数
split(A[0...n-1], leftCount(out), rightCount(out), modeCount(out))
if modeCount > maxCount:    // 如果大于之前的"众数"，则更新
  maxCount = modeCount
  finalMode = A[leftCount]  
if leftCount > modeCount:
  GetMode(A[0...leftCount-1], maxCount, finalMode)
if rightCount > modeCount:
  GetMode(A[leftCount+modeCount...n-1], maxCount, finalMode)
```

划分函数：

```
split(A[0...n-1], leftCount(out), rightCount(out), modeCount(out))
// 根据中轴划分数组A[0...n-1]
// 输入：数组A[0...n-1]
// 输出：左边元素的个数，右边元素的个数，中轴的个数
leftCount<-0;rightCount<-0;modeCount<-0
for i from 1 to n-1:
  if A[i] < A[0]:
    Acopy[leftCount] = A[i]
    leftCount<-leftCount+1
  else if A[i] > A[0]:
    Acopy[n-1-rightCount] = A[i]
    rightCount<-rightCount+1
  else:
    modeCount<-modeCount+1
```
