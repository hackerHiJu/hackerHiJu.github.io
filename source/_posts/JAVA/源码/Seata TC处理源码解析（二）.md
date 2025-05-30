---
title: Seata TC处理源码解析（二）
date: 2024-12-30 19:07:05
updated: 2024-12-30 19:07:05
tags:
  - Java
  - 源码
comments: true
categories:
  - Java
  - 源码
  - Seata
thumbnail: https://images.unsplash.com/photo-1682687982360-3fbab65f9d50?crop=entropy&cs=srgb&fm=jpg&ixid=M3w2NDU1OTF8MHwxfHJhbmRvbXx8fHx8fHx8fDE3NDMwODM1NjF8&ixlib=rb-4.0.3&q=85&w=1920&h=1080
published: true
---

# Seata TC处理源码解析

## 1. 服务启动

### 1.1 ServerRunner

Seata的TC服务启动代码是在 **ServerRunner** 中，实现了 **CommandLineRunner、DisposableBean** 两个接口，会在 **SpringBoot** 容器启动完成之后调用 **run()**，**DisposableBean** 会在容器关闭时调用 **distroy()**

```java
@Component
public class ServerRunner implements CommandLineRunner, DisposableBean {
    //是否已经启动
    private boolean started = Boolean.FALSE;
    @Override
    public void run(String... args) {
        try {
            long start = System.currentTimeMillis();
            //启动服务端
            Server.start(args);
            started = true;
            long cost = System.currentTimeMillis() - start;
            LOGGER.info("seata server started in {} millSeconds", cost);
        } catch (Throwable e) {
            started = Boolean.FALSE;
            LOGGER.error("seata server start error: {} ", e.getMessage(), e);
            System.exit(-1);
        }
    }
}
```

### 1.2 Server

TC的启动和创建方式跟TM、RM的代码一样 **nettyRemotingServer.init();** 方法会调用 **registerProcessor()** 注册上对应协议类型的处理器，这里用的比较多的是 **ServerOnRequestProcessor** 处理器

```java
public class Server {
    public static void start(String[] args) {
        // create logger
        final Logger logger = LoggerFactory.getLogger(Server.class);
        //解析启动参数
        ParameterParser parameterParser = new ParameterParser(args);

        //initialize the metrics
        //初始化监控管理器
        MetricsManager.get().init();

        //设置存储模式 file, db, redis
        System.setProperty(ConfigurationKeys.STORE_MODE, parameterParser.getStoreMode());

        //设置netty的工作线程池
        ThreadPoolExecutor workingThreads = new ThreadPoolExecutor(NettyServerConfig.getMinServerPoolSize(),
                NettyServerConfig.getMaxServerPoolSize(), NettyServerConfig.getKeepAliveTime(), TimeUnit.SECONDS,
                new LinkedBlockingQueue<>(NettyServerConfig.getMaxTaskQueueSize()),
                new NamedThreadFactory("ServerHandlerThread", NettyServerConfig.getMaxServerPoolSize()), new ThreadPoolExecutor.CallerRunsPolicy());
        //创建netty服务端
        NettyRemotingServer nettyRemotingServer = new NettyRemotingServer(workingThreads);
        //初始化服务节点生成的id方式
        UUIDGenerator.init(parameterParser.getServerNode());
        //log store mode : file, db, redis
        //日志存储的模式
        SessionHolder.init(parameterParser.getSessionStoreMode());
        //锁日志的存储模式
        LockerManagerFactory.init(parameterParser.getLockStoreMode());
        //获取到默认协调器，其中处理对应消息的入栈事件（目的是处理TM、RM发送过来的请求消息客户端使用的是 DefaultRMHandler ）
        DefaultCoordinator coordinator = DefaultCoordinator.getInstance(nettyRemotingServer);
        //初始化设置各个线程池 重试线程池、重试提交事务线程池、异步提交线程池、超时检测线程池、undolog日志删除线程池
        coordinator.init();
        //设置处理器，会在处理器当中进行调用 registerProcessor() 方法进行对不同消息类型的处理设置
        nettyRemotingServer.setHandler(coordinator);
        //将默认协调器，添加到生命周期监听器当中，当容器销毁时自动回收
        ServerRunner.addDisposable(coordinator);
        //绑定host的ip地址
        if (NetUtil.isValidIp(parameterParser.getHost(), false)) {
            XID.setIpAddress(parameterParser.getHost());
        } else {
            String preferredNetworks = ConfigurationFactory.getInstance().getConfig(REGISTRY_PREFERED_NETWORKS);
            if (StringUtils.isNotBlank(preferredNetworks)) {
                XID.setIpAddress(NetUtil.getLocalIp(preferredNetworks.split(REGEX_SPLIT_CHAR)));
            } else {
                XID.setIpAddress(NetUtil.getLocalIp());
            }
        }
        //注册处理器，跟客户端初始化一样，可以看 https://blog.csdn.net/weixin_43915643/article/details/127964285?spm=1001.2014.3001.5501 网络请求相关的处理
        nettyRemotingServer.init();
    }
}
```

### 1.3 ServerOnRequestProcessor

通过netty接收到通信，然后通过设置的处理器 **io.seata.core.rpc.netty.AbstractNettyRemotingServer.ServerHandler#channelRead** 根据类型选择对应的处理器调用到当前处理器的 **process()** 方法

- 分支注册
- 分支状态上报
- 开启全局事务
-  全局事务提交
- 全局锁的查询
- 全局事务回滚
- 全局事务的状态
- seata合并处理

```java
/**
     * 请求处理器，默认当前类会处理的消息类型：
     * 1.分支注册
     * 2.分支状态上报
     * 3.开启全局事务
     * 4.全局事务提交
     * 5.全局锁的查询
     * 6.全局事务回滚
     * 7.全局事务的状态
     * 8.seata合并处理
     *
     * @param ctx
     * @param rpcMessage
     * @throws Exception
     */
    @Override
    public void process(ChannelHandlerContext ctx, RpcMessage rpcMessage) throws Exception {
        //是否已经被注册了管道
        if (ChannelManager.isRegistered(ctx.channel())) {
            onRequestMessage(ctx, rpcMessage);
        } else {
            try {
                if (LOGGER.isInfoEnabled()) {
                    LOGGER.info("closeChannelHandlerContext channel:" + ctx.channel());
                }
                //断开链接
                ctx.disconnect();
                //关闭
                ctx.close();
            } catch (Exception exx) {
                LOGGER.error(exx.getMessage());
            }
            if (LOGGER.isInfoEnabled()) {
                LOGGER.info(String.format("close a unhandled connection! [%s]", ctx.channel().toString()));
            }
        }
    }

private void onRequestMessage(ChannelHandlerContext ctx, RpcMessage rpcMessage) {
        // 当前消息是否是批量发送的消息类型
        if (message instanceof MergedWarpMessage) {
            ........
            }
        } else {
            final AbstractMessage msg = (AbstractMessage) message;
            //使用事务消息处理器，服务端采用的 DefaultCoordinator，客户端采用的 DefaultRMHandler
            AbstractResultMessage result = transactionMessageHandler.onRequest(msg, rpcContext);
            remotingServer.sendAsyncResponse(rpcMessage, ctx.channel(), result);
        }
    }
```

### 1.4 DefaultCoordinator

TC端默认采用的消息处理器其中通过方法重载的方式通过不同的消息调用不同的 **handle()** 方法 

![1673228741345](images/1673228741345.png)

所有的请求体都继承至 **AbstractTransactionRequestToTC**，例如：BranchRegisterRequest，AbstractTransactionRequestToTC中定义了一个方法 **setTCInboundHandler()**，这个方法要求传入一个 TCInboundHandler接口类型，这个接口定义了所有消息类型的处理方法；这里调用的 **handle()** 是一个抽象方法通过子类来进行实现

```java
public AbstractResultMessage onRequest(AbstractMessage request, RpcContext context) {
        if (!(request instanceof AbstractTransactionRequestToTC)) {
            throw new IllegalArgumentException();
        }
        AbstractTransactionRequestToTC transactionRequest = (AbstractTransactionRequestToTC) request;
        //将当前处理类，DefaultCoordinator传入到消息实体中，AbstractMessage继承至 AbstractTransactionRequestToTC
        transactionRequest.setTCInboundHandler(this);

        //这里调用消息的 handle() 方法，就相当于调用 this.handle() 方法，这样的目的是可以通过根据不同的类型调用对应的handle()方法
        return transactionRequest.handle(context);
    }

//通过子类来实现，this代表的是请求方法体的类型，例如：BranchRegisterRequest，就会调用到 DefaultCoordinator.handle() 对应的类型方法中
public AbstractTransactionResponse handle(RpcContext rpcContext) {
    return handler.handle(this, rpcContext);
}
```

### 1.5 TCInboundHandler

```java
ublic interface TCInboundHandler {

    /**
     * 处理开启全局事务的请求
     */
    GlobalBeginResponse handle(GlobalBeginRequest globalBegin, RpcContext rpcContext);

    /**
     * 处理全局事务提交的请求
     */
    GlobalCommitResponse handle(GlobalCommitRequest globalCommit, RpcContext rpcContext);

    /**
     * 处理事务回滚的请求
     */
    GlobalRollbackResponse handle(GlobalRollbackRequest globalRollback, RpcContext rpcContext);

    /**
     * 处理分支注册的请求
     */
    BranchRegisterResponse handle(BranchRegisterRequest branchRegister, RpcContext rpcContext);

    /**
     * 处理分支状态上报的请求
     */
    BranchReportResponse handle(BranchReportRequest branchReport, RpcContext rpcContext);

    /**
     * 查询全局锁
     */
    GlobalLockQueryResponse handle(GlobalLockQueryRequest checkLock, RpcContext rpcContext);

    /**
     * 全局事务状态查询
     */
    GlobalStatusResponse handle(GlobalStatusRequest globalStatus, RpcContext rpcContext);

    /**
     * 上报事务状态该的请求
     */
    GlobalReportResponse handle(GlobalReportRequest globalReport, RpcContext rpcContext);

}
```



## 2. 一阶段

在处理时会通过 **Core** 接口的实现类 **DefaultCore** 进行处理，其中维护了一个 **Map<BranchType, AbstractCore\>** 的结构，通过不同的模式获取到不同的处理器

- ATCore
- XACore
- TccCore
- SagaCore

### 2.1 GlobalSession

全局会话实体，当前对象会在全局事务发起时创建

```java
public class GlobalSession implements SessionLifecycle, SessionStorable {
    //全局事务id
    private String xid;
    //事务的id
    private long transactionId;
    //事务的状态
    private volatile GlobalStatus status;
    //应用程序的id
    private String applicationId;
    //事务分组
    private String transactionServiceGroup;
    //事务名称
    private String transactionName;
    //事务超时时间
    private int timeout;
    //事务开启时间
    private long beginTime;
    //应用程序的元数据
    private String applicationData;
    //懒加载分支
    private final boolean lazyLoadBranch;
    //是否活跃
    private volatile boolean active = true;
    //分支会话
    private List<BranchSession> branchSessions;
    //全局会话锁对象使用的是 ReentrantLock
    private GlobalSessionLock globalSessionLock = new GlobalSessionLock();
}
```

### 2.2 BranchSession

分支会话，每一个事务分支都会创建一个会话

```java
public class BranchSession implements Lockable, Comparable<BranchSession>, SessionStorable {
    //全局事务id
    private String xid;
    //事务id
    private long transactionId;
    //分支id
    private long branchId;
    //资源组的id
    private String resourceGroupId;
    //资源id
    private String resourceId;
    //分支锁
    private String lockKey;
    //分支类型
    private BranchType branchType;
    //分支状态
    private BranchStatus status = BranchStatus.Unknown;
    //客户端id
    private String clientId;
    //应用数据
    private String applicationData;
    //分支锁的状态
    private LockStatus lockStatus = Locked;
    //锁持有器
    private ConcurrentMap<FileLocker.BucketLockMap, Set<String>> lockHolder
        = new ConcurrentHashMap<>();
}
```

### 2.3 SessionManager

会话管理器，这个接口主要用于对 **session** 会话的持久化，其中有三种类型

- DataBaseSessionManager
- FileSessionManager
- RedisSessionManager

这个类型的接口也继承了 **SessionLifecycleListener** 会话生命周期监听器，用于在各个阶段执行会话的回调处理

### 2.4 事务开始

#### 消息体

**io.seata.core.protocol.transaction.GlobalBeginRequest**

#### 方法

**io.seata.server.coordinator.DefaultCoordinator#doGlobalBegin**

```java
public String begin(String applicationId, String transactionServiceGroup, String name, int timeout)
        throws TransactionException {
        //根据应用名称、事务分组、事务名称、超时时间创建一个全局事务的会话
        GlobalSession session = GlobalSession.createGlobalSession(applicationId, transactionServiceGroup, name,
            timeout);
        MDC.put(RootContext.MDC_KEY_XID, session.getXid());
        /**
         * 会话添加一个生命周期监听器 SessionManager根据存储的类型通过spi机制加载出来
         * 目前有三个实现方式
         * FileSessionManager
         * RedisSessionManager
         * DataBaseSessionManager
         */
        session.addSessionLifecycleListener(SessionHolder.getRootSessionManager());

        /**
         * 设置事务状态为开始
         * 设置事务开始时间
         * 设置事务是否活跃
         * 调用生命周期监听函数 onBegin() 将对应的事务会话进行持久化
         */
        session.begin();

        //发布事务开始的事件
        MetricsPublisher.postSessionDoingEvent(session, false);

        // XID.generateXID(transactionId); 事务id的组成是 ip地址:端口:随机uuid
        return session.getXid();
    }
```

### 2.5 分支注册

#### 消息体

**io.seata.core.protocol.transaction.BranchRegisterRequest**

#### 方法

**io.seata.server.coordinator.DefaultCoordinator#doBranchRegister**

```java
protected void doBranchRegister(BranchRegisterRequest request, BranchRegisterResponse response,
                                    RpcContext rpcContext) throws TransactionException {
        MDC.put(RootContext.MDC_KEY_XID, request.getXid());
        //根据分支的模式，获取到对应的 Core 类型进行分支的注册，默认是 AT模式
        response.setBranchId(
                core.branchRegister(request.getBranchType(), request.getResourceId(), rpcContext.getClientId(),
                        request.getXid(), request.getApplicationData(), request.getLockKey()));
    }
```

```java
public Long branchRegister(BranchType branchType, String resourceId, String clientId, String xid,
                               String applicationData, String lockKeys) throws TransactionException {
        //先通过xid获取到全局session对象，然后后续的事务参与者进行上报时创建 BranchSession 对象存入GlobalSession中
        GlobalSession globalSession = assertGlobalSessionNotNull(xid, false);
        //根据对应的 SessionManager 类型存入构建好
        return SessionHolder.lockAndExecute(globalSession, () -> {
            //检查事务会话的状态
            globalSessionStatusCheck(globalSession);
            //添加生命周期监听器
            globalSession.addSessionLifecycleListener(SessionHolder.getRootSessionManager());
            //根据分支类型、资源id、使用数据、行锁、客户端id创建 分支session
            BranchSession branchSession = SessionHelper.newBranchByGlobal(globalSession, branchType, resourceId,
                    applicationData, lockKeys, clientId);
            MDC.put(RootContext.MDC_KEY_BRANCH_ID, String.valueOf(branchSession.getBranchId()));
            //通过对应的 LockManager 类型（RedisLockManager、DataBaseLockManager、FileLockManager）对分支记录进行加锁，如果加锁失败就直接抛出异常
            branchSessionLock(globalSession, branchSession);
            try {
                //将分支会话添加到全局会话中
                globalSession.addBranch(branchSession);
            } catch (RuntimeException ex) {
                branchSessionUnlock(branchSession);
                throw new BranchTransactionException(FailedToAddBranch, String
                        .format("Failed to store branch xid = %s branchId = %s", globalSession.getXid(),
                                branchSession.getBranchId()), ex);
            }
            if (LOGGER.isInfoEnabled()) {
                LOGGER.info("Register branch successfully, xid = {}, branchId = {}, resourceId = {} ,lockKeys = {}",
                        globalSession.getXid(), branchSession.getBranchId(), resourceId, lockKeys);
            }
            //返回分支会话的id，分支id，uuid
            return branchSession.getBranchId();
        });
    }
```

