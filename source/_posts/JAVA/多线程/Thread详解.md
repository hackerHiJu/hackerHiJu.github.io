---
title: Thread详解
date: 2024-12-30 19:07:05
updated: 2024-12-30 19:07:05
tags: ["Java", "多线程"]
comments: true
categories: ["Java"]
thumbnail: "https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/%25E5%25A4%25A9%25E7%25A9%25BA.png"
published: false
---

# Thread详解

**java.lang.Thread** 类

## 1. 核心属性

### 1.1 线程存活状态

通过 **isAlive()** 方法来判断一条线程的存活状态。一条线程的寿命起始于它真正的在 **start()** 方法中被启动起来，结束于它刚刚离开 **run()** 方法，此时线程死亡。

### 1.2 线程执行状态

- NEW：该状态下线程还没有开始执行
- RUNNABLE：该状态下线程正在JVM中执行
- BLOCKED：该状态下线程被阻塞并且等待一个监听锁
- WAITING：该状态下线程无限期等待另外一条线程执行特定的操作
- TIMED_WAITING：该状态下线程在特定的时间内等待另外一条线程执行某种操作
- TERMINATED：该状态下线程已经退出

```java
public static void main(String[] args) {
    Thread thread = new Thread(() -> {
        System.out.println("线程执行");
    });
    System.out.println(thread.getState());
    thread.start();
    System.out.println(thread.getState());
    try {
        thread.join();
    } catch (InterruptedException e) {
        e.printStackTrace();
    }
    System.out.println(thread.getState());
}
```

```text
NEW
RUNNABLE
线程执行
TERMINATED
```

### 1.3 线程优先级

一般情况下当计算机有足够的处理器或处理内核，操作系统就会为每个处理器或核心分配单独的线程，这些线程可以同时执行。一旦计算机没有足够的处理器或核心的时候，多条线程只能轮转着使用共享的处理器和核心

> Runtime.*getRuntime*().availableProcessors()  #可以获取当前计算器可用的处理器

操作系统使用调度器来决定什么时候来执行等待的线程

- Linux2.6到2.6.23使用的O(1)调度器，并且2.6.23也使用了Completely Fair调度器，它也是默认的调度器
- Windows基于NT的操作系统使用了多级反馈的队列调度器

