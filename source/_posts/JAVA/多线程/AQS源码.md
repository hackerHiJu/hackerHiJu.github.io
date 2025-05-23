---
title: AQS源码解析
date: 2024-12-30 19:07:05
updated: 2024-12-30 19:07:05
tags: ["Java", "多线程"]
comments: true
categories: ["Java"]
thumbnail: "https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/%25E5%25A4%25A9%25E7%25A9%25BA.png"
---

# 一、AQS

AbstractQueuedSynchronizer：抽象队列同步器

## 1、AQS具备的特性

- 阻塞等待队列
- 共享/独占
- 公平/非公平
- 可重入
- 允许中断

## 2、AQS内部实现

Java.concurrent.util当中同步器的实现如Lock,Latch,Barrier等，都是基于AQS框架实现

- 一般通过定义内部类Sync继承AQS
- 将同步器所有调用都映射到Sync对应的方法

## 3、AQS框架-管理状态

- AQS内部维护属性volatile int state (32位)：state表示资源的可用状态
- State三种访问方式
  - getState()
  - setState()
  - compareAndSetState()
- AQS定义两种资源共享方式
  - Exclusive：独占，只有一个线程能执行，如ReentrantLock
  - Share：共享，多个线程可以同时执行，如Semaphore/CountDownLatch
- AQS定义两种队列
  - 同步等待队列
  - 条件等待队列

## 4、CLH队列

基于双向链表实现数据结构的队列，先FIFO先进先出的线程等待队列，Java中的CLH队列是原CLH队列的一个变种，线程由原先的自旋改为阻塞机制

## 5、JUC编程Tools

#### （1）五个工具类

- Executors
- Semaphore（Node共享模式）
- Exchanger
- CyclicBarrier（Node共享模式）
- CountDownLatch（Node共享模式）

## 6、Atomic&Unsafe魔法类

#### （1）Atomic

- 基本类：AtomicInteger、AtomicLong、AtomicBoolean
- 引用类型：AtomicReference、AtomicReference的ABA实例、AtomicStampeRerence、AtomicMarkableReference
- 数组类型：AtomicIntegerArray、AtomicLongArray、AtomicReferenceArray
- 属性原子修改器（Updater）：AtomicIntegerFieldUpdater、AtomicLongFieldUpdater、AtomicReferenceFieldUpdater

都依赖于Unsafe魔术类，直接越过虚拟机对其进行操作

#### （2）ABA问题

过程无法跟踪

# 二、源码

## 1、ReentrantLock（悲观锁）

```java
private ReentrantLock lock = new ReentrantLock(true);
public class ReentrantLock implements Lock, java.io.Serializable {
	private final Sync sync;
    /**
    判断是公平锁还是非公平锁
    **/
    public ReentrantLock(boolean fair) {
      sync = fair ? new FairSync() : new NonfairSync();
    }
	//调用加锁的方法，实际上是通过 ReentrantLock内部维护的Sync对象来调用
    public void lock() {
      sync.lock();
    }
}
```

## 2、Node（CLH队列）

```java
static final class Node {
        /**
         * 标记节点未共享模式
         * */
        static final Node SHARED = new Node();
        /**
         *  标记节点为独占模式
         */
        static final Node EXCLUSIVE = null;

        /**
         * 在同步队列中等待的线程等待超时或者被中断，需要从同步队列中取消等待
         * */
        static final int CANCELLED =  1;
        /**
         *  后继节点的线程处于等待状态，而当前的节点如果释放了同步状态或者被取消，将会通知后继节点，使后继节点的线程得以运行。
         */
        static final int SIGNAL    = -1;
        /**
         *  节点在等待队列中，节点的线程等待在Condition上，当其他线程对Condition调用了signal()方法后，该节点会从等待队列中转移到同步队列中，加入到同步状态的获取中。存在于条件队列当中
         */
        static final int CONDITION = -2;
        /**
         * （如果是共享模式）如果是广播状态那么就会去唤醒下一个，如果下一个还是广播模式依然会被唤醒，直到下一个状态不是广播状态
         */
        static final int PROPAGATE = -3;

        /**
         * 标记当前节点的信号量状态 (1,0,-1,-2,-3)5种状态
         * 使用CAS更改状态，volatile保证线程可见性，高并发场景下，
         * 即被一个线程修改后，状态会立马让其他线程可见。
         */
        volatile int waitStatus;

        /**
         * 前驱节点，当前节点加入到同步队列中被设置
         */
        volatile Node prev;

        /**
         * 后继节点
         */
        volatile Node next;

        /**
         * 节点同步状态的线程
         */
        volatile Thread thread;

        /**
         * 等待队列中的后继节点，如果当前节点是共享的，那么这个字段是一个SHARED常量， 也就是说节点类型(独占和共享)和等待队列中的后继节点共用同一个字段。
         */
        Node nextWaiter;

        /**
         * Returns true if node is waiting in shared mode.
         */
        final boolean isShared() {
            return nextWaiter == SHARED;
        }

        /**
         * 返回前驱节点
         */
        final Node predecessor() throws NullPointerException {
            Node p = prev;
            if (p == null)
                throw new NullPointerException();
            else
                return p;
        }

        Node() {    // Used to establish initial head or SHARED marker
        }

        Node(Thread thread, Node mode) {     // Used by addWaiter
            this.nextWaiter = mode;
            this.thread = thread;
        }

        Node(Thread thread, int waitStatus) { // Used by Condition
            this.waitStatus = waitStatus;
            this.thread = thread;
        }
    }
```

## 3、FairSync（公平锁）

```java
static final class FairSync extends Sync {
        final void lock() {
            acquire(1);
        }
-------------------------------------------------
    public final void acquire(int arg) {
        //tryAcquire()方法会去尝试获取到锁的状态，如果获取失败，将线程加入到等待队列当中。addWaiter(),默认的模式是独占状态（排它锁）
        if (!tryAcquire(arg) &&
                acquireQueued(addWaiter(Node.EXCLUSIVE), arg))
            selfInterrupt();
    }
-------------------------------------------------
    //当节点被加入到等待队列之后，会重新进行获取锁
    final boolean acquireQueued(final Node node, int arg) {
        boolean failed = true;
        try {
            boolean interrupted = false;
            for (;;) {
                //获取到当前节点指向的前驱节点
                final Node p = node.predecessor();
                //对比前驱节点是否等于头节点，如果等于前驱节点，那么再此尝试获取锁
                if (p == head && tryAcquire(arg)) {
                    setHead(node);
                    p.next = null; // help GC
                    failed = false;
                    return interrupted;
                }
                if (shouldParkAfterFailedAcquire(p, node) &&
                    parkAndCheckInterrupt())
                    interrupted = true;
            }
        } finally {
            if (failed)
                cancelAcquire(node);
        }
    }
    
    
-------------------------------------------------
    //尝试获取锁
    protected final boolean tryAcquire(int acquires) {
            final Thread current = Thread.currentThread();
            int c = getState();
            if (c == 0) {
                //判断当前队列当中是否有等待的线程
                if (!hasQueuedPredecessors() &&
                    compareAndSetState(0, acquires)) {
                    //设置当前锁的拥有者
                    setExclusiveOwnerThread(current);
                    return true;
                }
            }
        //如果当前线程等于当前持有锁的线程，那么就将state的值进行+1
            else if (current == getExclusiveOwnerThread()) {
                int nextc = c + acquires;
                if (nextc < 0)
                    throw new Error("Maximum lock count exceeded");
                setState(nextc);
                return true;
            }
            return false;
        }
-------------------------------------------------
    //再获取锁之前先判断队列当中是否有等待状态的线程
    public final boolean hasQueuedPredecessors() {
        Node t = tail; 
        Node h = head;
        Node s;
        return h != t &&
            ((s = h.next) == null || s.thread != Thread.currentThread());
    }
    
-------------------------------------------------
    private Node addWaiter(Node mode) {
        //将当前线程构建成Node类型
        Node node = new Node(Thread.currentThread(), mode);
        Node pred = tail;
        // 当前尾节点是否为null？
        if (pred != null) {
            //将当前节点尾插入的方式
            node.prev = pred;
            //CAS将节点插入同步队列的尾部
            if (compareAndSetTail(pred, node)) {
                pred.next = node;
                return node;
            }
        }
        //当尾节点为空的时候调用
        enq(node);
        return node;
    }
-------------------------------------------------
    private Node enq(final Node node) {
        //死循环
        for (;;) {
            Node t = tail;
            if (t == null) {
                //队列为空需要初始化，创建空的头节点
                if (compareAndSetHead(new Node()))
                    tail = head;
            } else {
                node.prev = t;
                //set尾部节点
                if (compareAndSetTail(t, node)) {//当前节点置为尾部
                    t.next = node; //前驱节点的next指针指向当前节点
                    return t;
                }
            }
        }
    }
-------------------------------------------------
    //将当前线程阻塞
    static void selfInterrupt() {
        Thread.currentThread().interrupt();
    }
    //释放锁
    public final boolean release(int arg) {
        if (tryRelease(arg)) {
            Node h = head;
            if (h != null && h.waitStatus != 0)
                unparkSuccessor(h);
            return true;
        }
        return false;
    }
-------------------------------------------------
    protected final boolean tryRelease(int releases) {
        //将状态值减去1（如果多次加锁）
            int c = getState() - releases;
        //如果当前线程不等于锁的持有线程那么就抛出异常
            if (Thread.currentThread() != getExclusiveOwnerThread())
                throw new IllegalMonitorStateException();
            boolean free = false;
        //如果状态值等于0，将锁释放掉
            if (c == 0) {
                free = true;
                setExclusiveOwnerThread(null);
            }
            setState(c);
            return free;
        }
-------------------------------------------------
    //唤醒队列当中的线程
    private void unparkSuccessor(Node node) {
        int ws = node.waitStatus;
        if (ws < 0)
            compareAndSetWaitStatus(node, ws, 0);
        Node s = node.next;
        if (s == null || s.waitStatus > 0) {
            s = null;
            for (Node t = tail; t != null && t != node; t = t.prev)
                if (t.waitStatus <= 0)
                    s = t;
        }
        //如果下一个节点不为空，那么唤醒节点的线程
        if (s != null)
            LockSupport.unpark(s.thread);
    }
}
```

## 4、Semaphore

```java
public class SemaphoreSample {

    public static void main(String[] args) {
        Semaphore semaphore = new Semaphore(2);
        for (int i=0;i<5;i++){
            new Thread(new Task(semaphore,"yangguo+"+i)).start();
        }
    }

    static class Task extends Thread{
        Semaphore semaphore;

        public Task(Semaphore semaphore,String tname){
            this.semaphore = semaphore;
            this.setName(tname);
        }

        public void run() {
            try {
                semaphore.acquire();   		      	
                	System.out.println(Thread.currentThread().getName()+":aquire() at time:"+System.currentTimeMillis());

                Thread.sleep(1000);

                semaphore.release();
                System.out.println(Thread.currentThread().getName()+":aquire() at time:"+System.currentTimeMillis());
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

        }
    }
}
```

```java
//加锁
semaphore.acquire();

--------------------------------------------
    public final void acquireSharedInterruptibly(int arg)
            throws InterruptedException {
    //先判断当前线程是否已经中断，如果中断了就抛出异常
        if (Thread.interrupted())
            throw new InterruptedException();
    //尝试获取共享资源，如果返回的数据小于0，那么将线程存放到队列当中
        if (tryAcquireShared(arg) < 0)
            doAcquireSharedInterruptibly(arg);
    }
----------------------------------------------
    static final class NonfairSync extends Sync {
        private static final long serialVersionUID = -2694183684443567898L;

        NonfairSync(int permits) {
            super(permits);
        }
		//非公平锁调用父类方法
        protected int tryAcquireShared(int acquires) {
            return nonfairTryAcquireShared(acquires);
        }
    }
------------------------------------------------
    final int nonfairTryAcquireShared(int acquires) {
            for (;;) {//自旋获取当前共享资源
                int available = getState();
                int remaining = available - acquires;
                //如果小于0的话就直接返回，如果不小于执行CAS修改共享资源的数据
                if (remaining < 0 ||
                    compareAndSetState(available, remaining))
                    return remaining;
            }
        }
--------------------------------------------------
    private void doAcquireSharedInterruptibly(int arg)
        throws InterruptedException {
    //按照共享模式，将Node节点添加到队列当中
        final Node node = addWaiter(Node.SHARED);
        boolean failed = true;
        try {
            for (;;) {
                //获取当前节点的前驱节点
                final Node p = node.predecessor();
                if (p == head) {
                    //如果是头节点的话，将再此尝试获取共享资源
                    int r = tryAcquireShared(arg);
                    if (r >= 0) {
                        //获取到锁之后，清空头节点将当前节点设置为头节点，并且唤醒下一个节点
                        setHeadAndPropagate(node, r);
                        p.next = null; // help GC
                        failed = false;
                        return;
                    }
                }
                //
                if (shouldParkAfterFailedAcquire(p, node) &&
                    parkAndCheckInterrupt())
                    throw new InterruptedException();
            }
        } finally {
            if (failed)
                cancelAcquire(node);
        }
    }
-------------------------------------------------
    private Node addWaiter(Node mode) {
        Node node = new Node(Thread.currentThread(), mode);
    //获取等待队列的尾部节点，如果是第一次执行，那么tail就为null
        Node pred = tail;
        if (pred != null) {
            node.prev = pred;
            if (compareAndSetTail(pred, node)) {
                pred.next = node;
                return node;
            }
        }
    //将会执行初始化队列
        enq(node);
    //返回当前节点
        return node;
    }
---------------------------------------------------
    private Node enq(final Node node) {
        for (;;) { //自旋
            //获取尾部节点，第一次是为null，第二次循环tail指向的是头部节点
            Node t = tail;
            if (t == null) {
                //设置一个空的节点，将头部指向这个空节点
                if (compareAndSetHead(new Node()))
                    //并且将尾部节点指向头节点
                    tail = head;
            } else {
                //将传进来的前驱节点指向尾部节点（头节点）
                node.prev = t;
                if (compareAndSetTail(t, node)) {
                    //t这时是头节点，将头节点的后驱节点指向传进来的node
                    t.next = node;
                    return t;
                }
            }
        }
    }
-------------------------------------------------
    private void setHeadAndPropagate(Node node, int propagate) {
        Node h = head;
    //把当前节点全部清空，并且把当前节点设置为头节点
        setHead(node);
        if (propagate > 0 || h == null || h.waitStatus < 0 ||
            (h = head) == null || h.waitStatus < 0) {
            //获取下一个节点
            Node s = node.next;
            if (s == null || s.isShared())
                //唤醒线程，并且修改后续节点的状态
                doReleaseShared();
        }
    }
--------------------------------------------------
    private void doReleaseShared() {
        for (;;) {
            //获取到头节点
            Node h = head;
            if (h != null && h != tail) {
                int ws = h.waitStatus;
                //验证头节点的状态是否为待唤醒状态
                if (ws == Node.SIGNAL) {
                    if (!compareAndSetWaitStatus(h, Node.SIGNAL, 0))
                        continue; 
                    //唤醒线程
                    unparkSuccessor(h);
                }
                else if (ws == 0 &&
                         !compareAndSetWaitStatus(h, 0, Node.PROPAGATE))
                    continue;
            }
            if (h == head)                   
                break;
        }
    }
---------------------------------------------------
    private void unparkSuccessor(Node node) {
ils or if status is changed by waiting thread.
         //如果头节点的状态小于0，那么就将头节点的状态改为0
        int ws = node.waitStatus;
        if (ws < 0)
            compareAndSetWaitStatus(node, ws, 0);
    //获取到后驱节点
        Node s = node.next;
    //如果后续节点为空，那么把当前节点的前置节点指向当前
        if (s == null || s.waitStatus > 0) {
            s = null;
            for (Node t = tail; t != null && t != node; t = t.prev)
                if (t.waitStatus <= 0)
                    s = t;
        }
    //唤醒线程
        if (s != null)
            LockSupport.unpark(s.thread);
    }
--------------------------------------------------
    private static boolean shouldParkAfterFailedAcquire(Node pred, Node node) {
        int ws = pred.waitStatus;
    //若前驱结点的状态是SIGNAL，意味着当前结点可以被安全地park
        if (ws == Node.SIGNAL)
            return true;
        if (ws > 0) {
            do {
                node.prev = pred = pred.prev;
            } while (pred.waitStatus > 0);
            pred.next = node;
        } else {
            //当前驱节点waitStatus为 0 or PROPAGATE状态时；将其设置为SIGNAL状态，然后当前结点才可以被安全地park
            compareAndSetWaitStatus(pred, ws, Node.SIGNAL);
        }
        return false;
    }
```

```java
//解锁
public void release() {
        sync.releaseShared(1);
    }
-----------------------------------------
    public final boolean releaseShared(int arg) {
    //先尝试释放锁资源
        if (tryReleaseShared(arg)) {
            //
            doReleaseShared();
            return true;
        }
        return false;
    }

-----------------------------------------
    protected final boolean tryReleaseShared(int releases) {
            for (;;) { //自旋
                int current = getState();
                int next = current + releases;
                if (next < current) // overflow
                    throw new Error("Maximum permit count exceeded");
                //CAS设置共享锁资源
                if (compareAndSetState(current, next))
                    return true;
            }
        }
--------------------------------------------
    private void doReleaseShared() {
        for (;;) { //自旋
            Node h = head;
            if (h != null && h != tail) {
                int ws = h.waitStatus;
                /**
                如果头节点的状态为 SIGNAL状态
                将头节点waitStatus重置为0，unparkSuccessor(h)中如果waitStatus<0为设置为Node.PROPAGATE
                
                **/
                if (ws == Node.SIGNAL) {
                    if (!compareAndSetWaitStatus(h, Node.SIGNAL, 0))
                        //设置失败重新循环
                        continue; 
                    /**
                    head状态为SIGNAL并且成功设置为0之后，唤醒head.next的节点
                    **/
                    unparkSuccessor(h);
                }
                /**
                如果状态为0的话，将节点设置为PROPAGATE状态（传播状态），意味着将状态向后续节点传播
                **/
                else if (ws == 0 &&
                         !compareAndSetWaitStatus(h, 0, Node.PROPAGATE))
                    continue;
            }
            if (h == head) 
                break;
        }
    }
----------------------------------------------
    private void unparkSuccessor(Node node) {
    //如果节点状态小于0，那么会清楚掉当前状态
        int ws = node.waitStatus;
        if (ws < 0)
            compareAndSetWaitStatus(node, ws, 0);
    //唤醒下一个节点
        Node s = node.next;
    /**
    如果后继节点为空，或者后继节点为CANCEL状态，从尾部节点开始向前找正常阻塞状态的节点唤醒
    **/
        if (s == null || s.waitStatus > 0) {
            s = null;
            for (Node t = tail; t != null && t != node; t = t.prev)
                if (t.waitStatus <= 0)
                    s = t;
        }
        if (s != null)
            LockSupport.unpark(s.thread);
    }
```

## 5、CountDownLatch

### 5.1 特点

有主次之分，执行完之后，会在主线程等待合并；不能重复，共享资源在消耗完之后就没有了

## 6、CyclicBarrier

### 6.1 特点

可以重复，共享资源可以恢复

