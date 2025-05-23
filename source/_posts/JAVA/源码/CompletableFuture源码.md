---
title: CompletableFuture源码
date: 2024-12-30 19:07:05
updated: 2024-12-30 19:07:05
tags:
  - Java
  - 源码
comments: true
categories:
  - Java
  - 源码
  - 并发
thumbnail: https://images.unsplash.com/photo-1532187643603-ba119ca4109e?crop=entropy&cs=srgb&fm=jpg&ixid=M3w2NDU1OTF8MHwxfHJhbmRvbXx8fHx8fHx8fDE3NDMwODM1NTJ8&ixlib=rb-4.0.3&q=85&w=1920&h=1080
published: true
---

# CompletableFuture

## 1. 接口

- Future

```java
public interface Future<V> {

    /**
     * 尝试取消此任务，如果任务已经完成或者需求，或者由于其他原因无法取消此方法无效。
     * 返回值false，一般是因为已经完成了
     */
    boolean cancel(boolean mayInterruptIfRunning);

    /**
     * 如果此任务在正常完成之前被取消，则返回false
     */
    boolean isCancelled();

    /**
     * 任务是否已经完成
     */
    boolean isDone();

    /**
     * 等待任务完成后返回结果，在等待过程中可能任务已经被中断或者执行出现了异常
     */
    V get() throws InterruptedException, ExecutionException;

    /**
     * 有等待超时时间的任务返回值获取
     */
    V get(long timeout, TimeUnit unit)
        throws InterruptedException, ExecutionException, TimeoutException;
}
```



- CompletionStage：阶段性完成任务接口

```java
public interface CompletionStage<T> {

    /**
     * 任务正常执行完成后会执行传递进去的指定函数方法，然后返回指定的类型
     */
    public <U> CompletionStage<U> thenApply(Function<? super T,? extends U> fn);

    /**
     * 任务正常执行完成后会执行传递进去的指定函数方法，然后返回指定的类型（异步执行）
     */
    public <U> CompletionStage<U> thenApplyAsync
        (Function<? super T,? extends U> fn);

    /**
     * 指定线程池来执行任务正常执行完成后会执行传递进去的指定函数方法，然后返回指定的类型
     */
    public <U> CompletionStage<U> thenApplyAsync
        (Function<? super T,? extends U> fn,
         Executor executor);

    /**
     * 执行任务正常执行完成后会执行传递进去的指定函数方法（没有返回值）
     */
    public CompletionStage<Void> thenAccept(Consumer<? super T> action);

    /**
     * 执行任务正常执行完成后会执行传递进去的指定函数方法（没有返回值，并且异步执行）
     */
    public CompletionStage<Void> thenAcceptAsync(Consumer<? super T> action);

    /**
     * 指定线程池来执行任务正常执行完成后会执行传递进去的指定函数方法（无返回值）
     */
    public CompletionStage<Void> thenAcceptAsync(Consumer<? super T> action,
                                                 Executor executor);
    /**
     * 当任务完成后执行的任务
     */
    public CompletionStage<Void> thenRun(Runnable action);

    /**
     * 当任务完成后执行的任务（异步执行）
     */
    public CompletionStage<Void> thenRunAsync(Runnable action);

    /**
     * 当任务完成后执行的任务（指定线程池执行）
     */
    public CompletionStage<Void> thenRunAsync(Runnable action,
                                              Executor executor);

    /**
     * 当前阶段和另外一个阶段的任务都完成时会执行当前阶段。两个阶段执行的结果会当成对应的参数传入，有返回值
     */
    public <U,V> CompletionStage<V> thenCombine
        (CompletionStage<? extends U> other,
         BiFunction<? super T,? super U,? extends V> fn);

    /**
     * 当前阶段和另外一个阶段的任务都完成时会执行当前阶段（异步执行）
     */
    public <U,V> CompletionStage<V> thenCombineAsync
        (CompletionStage<? extends U> other,
         BiFunction<? super T,? super U,? extends V> fn);

    /**
     * 当前阶段和另外一个阶段的任务都完成时会执行当前阶段（异步指定线程执行）
     */
    public <U,V> CompletionStage<V> thenCombineAsync
        (CompletionStage<? extends U> other,
         BiFunction<? super T,? super U,? extends V> fn,
         Executor executor);

    /**
     * 当前阶段和另外一个阶段的任务都完成时会执行当前阶段（无返回结果值）
     */
    public <U> CompletionStage<Void> thenAcceptBoth
        (CompletionStage<? extends U> other,
         BiConsumer<? super T, ? super U> action);

    /**
     * 当前阶段和另外一个阶段的任务都完成时会执行当前阶段（异步执行，无返回结果值）
     */
    public <U> CompletionStage<Void> thenAcceptBothAsync
        (CompletionStage<? extends U> other,
         BiConsumer<? super T, ? super U> action);

    /**
     * 当前阶段和另外一个阶段的任务都完成时会执行当前阶段（异步指定线程执行，无返回结果值）
     */
    public <U> CompletionStage<Void> thenAcceptBothAsync
        (CompletionStage<? extends U> other,
         BiConsumer<? super T, ? super U> action,
         Executor executor);

    /**
     * 当前阶段和给定的阶段都完成时才执行 action任务
     */
    public CompletionStage<Void> runAfterBoth(CompletionStage<?> other,
                                              Runnable action);
    /**
     * 当前阶段和给定的阶段都完成时才以异步的方式执行 action任务
     */
    public CompletionStage<Void> runAfterBothAsync(CompletionStage<?> other,
                                                   Runnable action);

    /**
     * 当前阶段和给定的阶段都完成时才以指定的线程池的方式执行 action任务
     */
    public CompletionStage<Void> runAfterBothAsync(CompletionStage<?> other,
                                                   Runnable action,
                                                   Executor executor);
    /**
     * 
     */
    public <U> CompletionStage<U> applyToEither
        (CompletionStage<? extends T> other,
         Function<? super T, U> fn);

    /**
     * 
     */
    public <U> CompletionStage<U> applyToEitherAsync
        (CompletionStage<? extends T> other,
         Function<? super T, U> fn);

    /**
     * 
     */
    public <U> CompletionStage<U> applyToEitherAsync
        (CompletionStage<? extends T> other,
         Function<? super T, U> fn,
         Executor executor);

    /**
     * 
     */
    public CompletionStage<Void> acceptEither
        (CompletionStage<? extends T> other,
         Consumer<? super T> action);

    /**
     * 
     */
    public CompletionStage<Void> acceptEitherAsync
        (CompletionStage<? extends T> other,
         Consumer<? super T> action);

    /**
     * 
     */
    public CompletionStage<Void> acceptEitherAsync
        (CompletionStage<? extends T> other,
         Consumer<? super T> action,
         Executor executor);

    /**
     * 
     */
    public CompletionStage<Void> runAfterEither(CompletionStage<?> other,
                                                Runnable action);

    /**
     * 
     */
    public CompletionStage<Void> runAfterEitherAsync
        (CompletionStage<?> other,
         Runnable action);

    /**
     * 
     */
    public CompletionStage<Void> runAfterEitherAsync
        (CompletionStage<?> other,
         Runnable action,
         Executor executor);

    /**
     * 
     */
    public <U> CompletionStage<U> thenCompose
        (Function<? super T, ? extends CompletionStage<U>> fn);

    /**
     * 
     */
    public <U> CompletionStage<U> thenComposeAsync
        (Function<? super T, ? extends CompletionStage<U>> fn);

    /**
     * 
     */
    public <U> CompletionStage<U> thenComposeAsync
        (Function<? super T, ? extends CompletionStage<U>> fn,
         Executor executor);

    /**
     * 
     */
    public <U> CompletionStage<U> handle
        (BiFunction<? super T, Throwable, ? extends U> fn);

    /**
     * 
     */
    public <U> CompletionStage<U> handleAsync
        (BiFunction<? super T, Throwable, ? extends U> fn);

    /**
     * 
     */
    public <U> CompletionStage<U> handleAsync
        (BiFunction<? super T, Throwable, ? extends U> fn,
         Executor executor);

    /**
     * 当任务执行完成时执行当前任务
     */
    public CompletionStage<T> whenComplete
        (BiConsumer<? super T, ? super Throwable> action);

    /**
     * 当任务执行完成时异步执行当前任务
     */
    public CompletionStage<T> whenCompleteAsync
        (BiConsumer<? super T, ? super Throwable> action);

    /**
     * 当任务执行完成时通过指定线程池异步执行当前任务
     */
    public CompletionStage<T> whenCompleteAsync
        (BiConsumer<? super T, ? super Throwable> action,
         Executor executor);

    /**
     * 当出现异常时会调用当前指定的函数
     */
    public CompletionStage<T> exceptionally
        (Function<Throwable, ? extends T> fn);

    /**
     * 将当前任务的类型转换为CompletableFuture任务
     */
    public CompletableFuture<T> toCompletableFuture();

}
```

- Completion：

  

## 2. 内部类

```java
abstract static class Completion extends ForkJoinTask<Void>
        implements Runnable, AsynchronousCompletionTask {
      volatile Completion next;      // 下一个任务

      /**
       * Performs completion action if triggered, returning a
       * dependent that may need propagation, if one exists.
       *
       * @param mode SYNC, ASYNC, or NESTED
       */
      abstract CompletableFuture<?> tryFire(int mode);

      /** Returns true if possibly still triggerable. Used by cleanStack. */
      abstract boolean isLive();

      public final void run()                { tryFire(ASYNC); }
      public final boolean exec()            { tryFire(ASYNC); return false; }
      public final Void getRawResult()       { return null; }
      public final void setRawResult(Void v) {}
  }
```



### 2.1 AsyncRun

异步执行器

```java
static final class AsyncRun extends ForkJoinTask<Void>
        implements Runnable, AsynchronousCompletionTask {
  		//获取任务执行结果的的编排任务
  		CompletableFuture<Void> dep;
  		
  		//真正需要执行的任务线程
  		Runnable fn;
  
  		AsyncRun(CompletableFuture<Void> dep, Runnable fn) {
          this.dep = dep; 
        	this.fn = fn;
      }
  
  		public void run() {
            CompletableFuture<Void> d; 
        		Runnable f;
        		//当编排的任务和目标需要执行的任务都不会空时进行执行
            if ((d = dep) != null && (f = fn) != null) {
                dep = null; fn = null;
                if (d.result == null) {
                    try {
                        f.run();
                        d.completeNull();
                    } catch (Throwable ex) {
                        d.completeThrowable(ex);
                    }
                }
                d.postComplete();
            }
        }
}
```



### 2.1 AsyncSupply

```java
static final class AsyncSupply<T> extends ForkJoinTask<Void>
        implements Runnable, AsynchronousCompletionTask {
}
```



## 3. 属性

```java
volatile Object result;       // 当前任务完成的返回值
volatile Completion stack;	  // 栈顶任务
```



## 4. 方法
