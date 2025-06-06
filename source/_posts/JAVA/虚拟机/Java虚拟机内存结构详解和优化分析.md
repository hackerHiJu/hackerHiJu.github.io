---
title: Java虚拟机内存结构详解和优化分析
date: 2025-04-01 02:12:33
updated: 2025-04-01 02:12:33
tags:
  - JVM
comments: true
categories: JVM
thumbnail: https://images.unsplash.com/photo-1682685796444-acc2f5c1b7b6?crop=entropy&cs=srgb&fm=jpg&ixid=M3w2NDU1OTF8MHwxfHJhbmRvbXx8fHx8fHx8fDE3NDM0ODc5NTN8&ixlib=rb-4.0.3&q=85&w=1920&h=1080
published: true
---
# JVM
一种可以运行Java字节码的虚拟机。只要编译文件符合虚拟机编译的文件格式要求。

## 一、java加载阶段

- 加载：载入字节码文件，只有使用到才会加载字节码文件
- 验证：校验字节文件的正确性
- 准备：给类的静态变量分配内存，并且赋予默认值
- 解析：将符号引用转换为直接引用
- 初始化：对类的静态变量初始化为指定的值
- 使用
- 卸载

## 二、类加载器的种类

- 启动类加载器（Bootstrap ClassLoader）:加载jar的核心类库，rt下面，c语言编写
- 扩展类加载器（Extension ClassLoader）：负责加载JRE扩展目录ext中的jar包
- 应用类加载器（Application ClassLoader）：负责加载ClassPath路径下的类包
- 用户自定义加载器（User ClassLoader）：负责加载用户自定义路径下的类包
- 全盘负责委托机制：当加载一个类当中引用了其它类时 也由这个当前类的加载器来加载

## 三、双亲委派机制（父类委派模型）

### 3.1 **如何判断是否是同一个类？**

内部还会判断类加载器是否一样

### 3.2 **为什么要设计双亲委派机制？**

- 沙箱安全机制：自己写的java.lang.String.class类不会被加载，这样便可以防止核心API库被随意更改
- 避免类的重复加载：当父类已经加载了该类时，就没有必要ClassLoader再加载一次，保证加载类的唯一性

### 3.3 **如何打破双亲委派机制？**

通过自定义类加载器，不要去委托父类进行加载。重写classLoad()方法，修改掉自带的双亲委派机制逻辑

## 四、java指令以及工具

- javap -c class文件 > text.txt 反编译到txt文件
- jinfo -flags 进程号 查看jvm的参数
- jstat -gc 进程号 查看各个空间的内存信息
- jmap -dump:format=b,file=temp.hprof 导出dump堆内存的快照 jdk自带的 jvisual工具
- jstack 进程号 > .txt文件中 查看线程
- jps查看java实现

## 五、jvm内存划分

### 5.1 栈

- 局部变量表：存放局部变量以及引用数据类型的地方
- 操作数栈：真正执行操作的地方
- 动态链接：在程序运行时，将符号引用转换为直接引用。符号引用就是一堆字符串
- 方法出口：方法指向调用当前方法的对象

### 5.2 方法区

静态变量+变量+类信息（构造方法/接口定义）+运行时常量池都存在方法区中

### 5.3  堆（Heap）

调优主要调 堆以及方法区
```java
优化方法：
1、System.gc()的调用：此方法是建议JVM进行Full GC，但是只是建议而非一定。可以使用-XX:+DisableExplicitGC来禁止RMI调用System.gc
2、老年代空间不足，经过Full GC之后依然不足，OutOfMemoryError错误，尽量保证对象在Minor GC时就被收集

新生代（1/3）：Young GC（Minor GC）
    Eden：内存比较小
    From：（如果满了就会引发一次轻GC）
    To：
    From如果内存满了就会将对象转移到To当中，然后To就变成From，From就变成了To相互进行转换，如果对象在From到To互相转换了15（转换次数可以设置）次 对象就会进入老年代区域（Full GC）
老年代（2/3内存）Full GC（Major GC）：存放比较大的对象以及存在时间比较久
元空间（方法区的实现）：动态扩展 jdk1.8

常用参数命令 
-XX:Xmx -XX:Xms 设置虚拟机最大最小，当内存达到最小值时会发生一次Full GC并且对内存进行扩容
-Xmn 设置整个新生代大小 
-XX:NewRatio  占比1:4，新生代占整个堆的1/5
-XX:SurvivorRatio 幸存代 设置Survivor:eden=2:8，即Survivor占年轻代的1/10
```

## 六、GC算法和收集器

- 如何判断对象可以被回收：
    - 引用计数器法：（每引用一次，就会+1）
        缺点：a引用b、b引用a 导致永远不会失去引用
    - 可达性分析法：（GC Roots，会去搜索从GCroots的对象作为起来，开始向下搜索节点所走过的路径）
        GC Roots根节点：类加载器、Thread、虚拟机栈的局部变量表、static成员、变量引用、本地方法栈的变量等
- 如何判断常量是一个废弃常量：
    如果局部变量表中没有引用常量池中的变量，那么就为废弃常量

## 七、如何判断一个类是无用的类：

- java堆中不存在该类的任何实例
- 加载该类的ClassLoader已经被回收
- 该类的Class对象没有在任何地方被引用，无法在任何地方通过反射访问该类

如果一个类没有用，但是又想让它不被回收，可以重写finalize方法并且引用自身对象来进行自救，只能自救一次

## 八、垃圾回收算法

- 复制算法：为了解决效率问题，将内存分为大小相同的两块，有一块保留内存，内存整理时将存活对象放到保留内存中，然后将使用的区域进行整理。缺点：只能存放较小的对象比如（From区域和To区域）
- 标记-清除算法：对可回收的内存进行标记，标记之后再清除 --效率很低，会出现内存碎片
- 标记-整理算法：先标记可回收的内存，依然还是需要每一个空间都要判断
- 分代收集算法：根据老年代还是新生代，使用不同的垃圾收集器

## 九、垃圾收集器（JDK1.8使用的Parallel并行收集器）

STW：停止所有的用户线程来跑垃圾收集

- Serial：停顿时间比较长
- ParNew：Serial收集器的多线程版本，停顿时间比较长，新生代用标记复制算法，老年代用户标记整理算法
- Parallel Scavenge：并行，多条垃圾收集器一起执行
- CMS（重点）：并发垃圾收集器。
    - 初始标记阶段（标记）
    - 并发标记（标记用户线程运行期间产生的垃圾）
    - 重新标记（重新进行一次标记）
    - 并发清除（并发的清除已经标记的垃圾）
    - 优点：STW的时间比较短
    - 缺点：对CPU资源敏感，无法处理浮动垃圾，使用的标记-清除算法会有大量空间碎片产生
- Serial Old、Parallel Old：根据老年代做出来的
- G1：
    - 初始标记
    - 并发标记
    - 最终标记
    - 筛选回收：可预测的停顿，独有的GC（MixGC），Full GC一般在内存溢出了。
```
将整个java堆分配成了多个大小相等的独立区域（Region），新生代和老年代只是一个部分Region的集合，默认分成2048个分区，可以通过-XX:G1HeapRegionSize=n来设置指定分区大小必须是2的幂。
每个Region又分成若干个大小为512byte的Card，-Xms/-Xmx设置堆的大小
```

## 十、怎么选择垃圾收集器（优先推荐G1）

- 优先调整堆的大小让服务器自己选择
- 如果内存小于100m，使用串行
- 如果是单核，没有停顿时间的要求，串行或者JVM自己选择
- 如果允许停顿时间不超过1秒，选择并行或者JVM自己选择
- 如果响应时间最重要，并且不能超过1秒，使用并发收集器

## 十一、调优，JVM调优主要调下面的两个指标

- 停顿时间：垃圾收集器做垃圾回收中断应用执行的时间。-XX:MaxGCPauseMills
- 吞吐量：垃圾收集的时间和总时间的占比:n/(1+n)，吞吐量为1-1/（1+n）。-XX:GCTimeRatio=n
```
例如：-XX:GCTimeRatio=9 默认是99 要求应用线程在整个执行时间中占9/10，其余1/10才是GC时间
```
## 十二、GC调优步骤

```
1、打印日志
-XX:+PrintGCDetails -XX:+PrintGCTimeStamps -XX:+PrintGCDateStamps -Xloggc:./gc.log
2、分析日志得到关键性指标
3、分析GC原因，调优JVM参数

Parallel：
    调优步骤一：分析日志造成YGC和Full GC比较的多的地方是哪里，元空间Full GC是因为加载的类太多，可以增大元空间的大小 -XX:MaxMetaspaceSize=?m、-XX:MetaspaceSize=?m
    调优步骤二：增大年轻代动态扩容增量（默认20%），减少YCG：-XX:YoungGenerationSizeIncrement=30
G1：
    步骤一：调整Region，如果出现YGC以及Mixed GC的话可以调整大小
    步骤二：调整GC与应用线程消耗时间的占比-XX:GCTimeRatio=99
    步骤三：-XX:MaxGCPauseMillis 默认200ms
```