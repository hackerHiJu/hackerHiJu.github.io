---
title: Collections&Queue体系
date: 2024-12-30 19:07:05
updated: 2024-12-30 19:07:05
tags: ["Java", "多线程"]
comments: true
categories: ["Java"]
thumbnail: "https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/%25E5%25A4%25A9%25E7%25A9%25BA.png"
published: false
---

# Collections&Queue体系

## 一、BlockingQueue

使用ReentrantLock，使用是条件队列condition（只能在独占模式下使用）

### 1.1 ArrayBlockingQueue

特点：数组支持的有界队列，如果初始化为1，不能扩容，只能等队列被消费了之后，才能继续添加，如果满了会将线程阻塞

条件队列（Condition）：

- notFull：将线程放入条件队列中，直到队列不满为止
- notEmpty：将线程放入条件队列中，直到队列不为空

### 1.2 LinkedBlockingQueue

链接节点支持的可选有界队列

### 1.3 PriorityBlockingQueue

优先级支持的无界优先级队列

### 1.4 DelayQueue

优先级堆支持的、基于时间的调度队列

## 二、HashMap

### 2.1 jdk1.7死锁

基础数据模型：数组+链表

原因：多线程情况下扩容期间，存在节点位置互换指针引用的问题
