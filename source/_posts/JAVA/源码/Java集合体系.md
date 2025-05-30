---
title: Java集合体系
date: 2024-12-30 19:07:05
updated: 2024-12-30 19:07:05
tags:
  - Java
  - 源码
comments: true
categories:
  - Java
  - 源码
  - 集合
thumbnail: https://images.unsplash.com/photo-1734552451995-75b247457544?crop=entropy&cs=srgb&fm=jpg&ixid=M3w2NDU1OTF8MHwxfHJhbmRvbXx8fHx8fHx8fDE3NDMwODM1MzF8&ixlib=rb-4.0.3&q=85&w=1920&h=1080
---


# java集合

## 1、架构图

![image-20210419160625571](images/image-20210419160625571.png)

- Iterator和ListIterator
- 集合框架分为Collection和Map两种，在两种框架体系下派生出：Map、Set、List、Queue
  - Map：存储Key，Value
  - Set：存储一系列不可重复对象，无序集合
  - List：可重复对象，有序集合，对象按头插入
  - Queue：队列容器，特性与List相似，只能从对头和对位操作元素
- JDK提供了Collections和Arrays工具类

## 2、Iterator、Iterable、ListIterator

- Iterator接口：
  - hasNext()：判断集合中是否有下一个对象
  - next()：返回集合中的下一个对象，并且将指针移动一位
  - remove()：删除集合中的对象

```java
public interface Iterator<E> {
    boolean hasNext();
    E next();
    void remove();
}
```

- Iterable接口：
  - iterator():提供了iterator接口
  - jdk1.8中，提供 forEach方法，使用了语法糖，本质还是采用的iterator进行遍历

```java
public interface Iterable<T> {
    Iterator<T> iterator();
    // JDK 1.8
    default void forEach(Consumer<? super T> action) {
        Objects.requireNonNull(action);
        for (T t : this) {
            action.accept(t);
        }
    }
}
```

![image-20210419161825390](images/image-20210419161825390.png)

为什么需要设计Iterator和Iterable？

Iterator的存在可以让子类去具体实现自己的迭代器，而Iterable的设计，重点在于forEach方法，提供了一个统一的迭代器。

- ListIterable接口：

  - 支持任意索引下标开始遍历，支持双向遍历

  ```java
  public interface ListIterator<E> extends Iterator<E> {
      boolean hasNext();
      E next();
      boolean hasPrevious();
      E previous();
      int nextIndex();
      int previousIndex();
      void remove();
      // 替换当前下标的元素,即访问过的最后一个元素
      void set(E e);
      void add(E e);
  }
  ```



## 3、Map和Collection接口

Collection存储元素本身，Map存储Key、Value，通过Key值进行映射。而有一部分采用了Map的一些特性进行实现。

例如：HashSet底层使用了HashMap，TreeSet底层使用TreeMap，LinkedHashSet底层使用LinkedHashMap

![image-20210419162636357](images/image-20210419162636357.png)

- Map接口下会将存储的方式细分为不同种类：
  - SoredMap接口：可以对<key,value>按照规则进行排序，具体实现有TreeMap
  - AbstractMap：为字类提供了通用的api实现
- Collection接口，提供了所有集合的通用方法（不包括Map）
  - 添加：add/addAll
  - 删除：remove/removeAll
  - 查找方法：contains/containsAll
  - 查询集合自身信息：size
- Collecttion下细分成不同的种类：
  - Set：不允许重复元素的无需集合，具体实现，HashSet/TreeSet
  - List：可存储重复元素的有序集合，具体实现，ArrayList/LinkedList
  - Queue：可存储重复元素的队列，具体实现。PriorityQueue / ArrayDeque

## 4、Map集合体系详解

Map体系下主要分为AbstractMap和SortedMap两类集合：

- AbstractMap：定义了普通map的通用行为，避免字类重复编写大量相同的代码，子类继承后可以重写方法，实现额外的逻辑
- SoredMap：定义了Map的排序行为，在内部定义好了有关排序的抽象方法，字类实现它时，必须重写所有方法

![image-20210419164039804](images/image-20210419164039804.png)

### 4.1 HashMap

通用的利用哈希表存储元素的集合，将元素放入HashMap时，计算哈希值转换为索引下标确定存放位置，查找时根据key的hash地址转换成数组的下标索引

HashMap底层是利用 数组+链表+红黑树 三种数据结构实现，**非线程安全**。

![image-20210419164302678](images/image-20210419164302678.png)

发生hash冲突时，会将相同映射地址的元素连成一条链表，如果链表超过8个时且数组元素超过了64，就会转换成为红黑树的结构。如果链表小于6个时就会转换为链表

### 4.2 为什么会将链表转换设置为8

hash 计算的结果离散好的话，那么红黑树这种形式是很少会被用到的，因为各个值都均匀分布，很少出现链表很长的情况。在理想情况下，链表长度符合泊松分布，各个长度的命中概率依次递减，当长度为 8 的时候，概率仅为 0.00000006。这是一个小于千万分之一的概率，通常我们的 Map 里面是不会存储这么多的数据的，所以通常情况下，并不会发生从链表向红黑树的转换。

### 4.3 HashMap死锁



## 5、LinkedHashMap

可以看作HashMap和LinkedList的结合：在HashMap的基础上添加了一条双向链表，由于这条双向链表使得LinkedHashMap可以实现LRU缓存淘汰策略，因为我们可以设置这条双向链表按照元素的访问次序进行排序

![image-20210419171014372](images/image-20210419171014372.png)