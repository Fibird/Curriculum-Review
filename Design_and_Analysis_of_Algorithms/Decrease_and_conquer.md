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
