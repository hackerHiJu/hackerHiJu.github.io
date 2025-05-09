---
title: Seata AT模式下的源码解析（一）
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
thumbnail: https://images.unsplash.com/photo-1581093691829-83572c1a4dec?crop=entropy&cs=srgb&fm=jpg&ixid=M3w2NDU1OTF8MHwxfHJhbmRvbXx8fHx8fHx8fDE3NDMwODM1NDN8&ixlib=rb-4.0.3&q=85&w=1920&h=1080
published: true
---

# Seata AT模式下的源码解析

源码：https://gitee.com/haijun1998/seata.git 分支source-read-1.5.0

## 1. GlobalTransactional

**@GlobalTransactional** 注解，提供给客户端来创建一个全局事务，**@GlobalTransactional** 注解由 **GlobalTransactionScanner** 进行扫描， 通过 **GlobalTransactionalInterceptor** 对其进行拦截增强

```java
@Retention(RetentionPolicy.RUNTIME)
@Target({ElementType.METHOD,ElementType.TYPE})
@Inherited
public @interface GlobalTransactional {

    /**
     * 全局事务超时时间，如果配置了 client.tm.default-global-transaction-timeout，将会替换默认值
     * 默认时间：60000毫秒
     */
    int timeoutMills() default DefaultValues.DEFAULT_GLOBAL_TRANSACTION_TIMEOUT;

    /**
     * 设置全局事务的名称
     */
    String name() default "";

    /**
     * 指定回滚的的异常类
     */
    Class<? extends Throwable>[] rollbackFor() default {};

    /**
     * 指定事务回滚的class名称
     */
    String[] rollbackForClassName() default {};

    /**
     * 指定不需要回滚的异常
     */
    Class<? extends Throwable>[] noRollbackFor() default {};

    /**
     * 指定不需要回滚的异常名称
     */
    String[] noRollbackForClassName() default {};

    /**
     * 事务的传播机制
     */
    Propagation propagation() default Propagation.REQUIRED;

    /**
     * 全局锁的重试间隔时间，默认0的话使用全局配置
     */
    int lockRetryInterval() default 0;

    /**
     * customized global lock retry interval(unit: ms)
     * you may use this to override global config of "client.rm.lock.retryInterval"
     * note: 0 or negative number will take no effect(which mean fall back to global config)
     *
     * @return int
     */
    @Deprecated
    @AliasFor("lockRetryInterval")
    int lockRetryInternal() default 0;

    /**
     * 全局锁获取的重试次数，0或者-1不生效，使用默认全局配置
     */
    int lockRetryTimes() default -1;
}
```

## 2. GlobalTransactionScanner

全局事务注解扫描器，继承了 **AbstractAutoProxyCreator** 用于在 **bean** 对象在创建时，对打了 **@GlobalTransactional** 注解的类添加 **Aop** 支持；

![](images/GlobalTransactionScanner.png)

其中 **GlobalTransactionScanner** 实现了 **wrapIfNecessary()** 方法；这个方法是在 **spring** 初始化完毕之后通过 **postProcessAfterInitialization()** 中进行调用，在 **spring aop** 进行增强时所用到的两个接口：

这里会在 **bean** 初始化之后调用 **wrapIfNecessary()** 方法对 **bean** 添加 **aop** 切面进行增强

```java
public interface BeanPostProcessor {
    //在spring调用 InitializingBean接口、自定义初始化方法之前执行
    default Object postProcessBeforeInitialization(Object bean, String beanName) throws BeansException {
        return bean;
    }

    //在spring调用 InitializingBean接口、自定义初始化方法之后执行
    @Nullable
    default Object postProcessAfterInitialization(Object bean, String beanName) throws BeansException {
        return bean;
    }
}
```

```java
public interface InstantiationAwareBeanPostProcessor extends BeanPostProcessor {
    //在执行创建bean之前执行
    @Nullable
    default Object postProcessBeforeInstantiation(Class<?> beanClass, String beanName) throws BeansException {
        return null;
    }
	
    //在执行 postProcessBeforeInstantiation() 方法如果返回值为null时执行
    default boolean postProcessAfterInstantiation(Object bean, String beanName) throws BeansException {
        return true;
    }

    //处理 PropertyValues，对bean通过 PropertyValues 类型进行处理，通过属性注入进行实现
    default PropertyValues postProcessProperties(PropertyValues pvs, Object bean, String beanName) throws BeansException {
        return null;
    }
}
```

### wrapIfNecessary()

其中会判断当前采用的事务模式，目前就是 **TCC、AT**

```java
protected Object wrapIfNecessary(Object bean, String beanName, Object cacheKey) {
        // 检查当前bean是否是 FactoryBean类型，是否已经被代理过了，是否是需要剔除的bean
        if (!doCheckers(bean, beanName)) {
            return bean;
        }
        try {
            //加锁防止多线程问题
            synchronized (PROXYED_SET) {
                //如果存在代理类中直接返回
                if (PROXYED_SET.contains(beanName)) {
                    return bean;
                }
                interceptor = null;
                /**
             * 根据4个类来进行解析 TCC 的调用方式
             * io.seata.rm.tcc.remoting.parser.DubboRemotingParser：是否是dubbo调用远程的方法
             * io.seata.rm.tcc.remoting.parser.LocalTCCRemotingParser：是否具有 @LocalTCC 注解
             * io.seata.rm.tcc.remoting.parser.SofaRpcRemotingParser
             * io.seata.rm.tcc.remoting.parser.HSFRemotingParser
             * 根据 @TwoPhaseBusinessAction 注解注册对应的方式到资源管理器中 （DefaultResourceManager）
             */
                if (TCCBeanParserUtils.isTccAutoProxy(bean, beanName, applicationContext)) {
                    // init tcc fence clean task if enable useTccFence
                    TCCBeanParserUtils.initTccFenceCleanTask(TCCBeanParserUtils.getRemotingDesc(beanName), applicationContext);
					
                    interceptor = new TccActionInterceptor(TCCBeanParserUtils.getRemotingDesc(beanName));
                    ConfigurationCache.addConfigListener(ConfigurationKeys.DISABLE_GLOBAL_TRANSACTION,
                            (ConfigurationChangeListener)interceptor);
                } else {
                    //获取到当前bean的class类，如果是代理对象需要获取到目标类型
                    Class<?> serviceInterface = SpringProxyUtils.findTargetClass(bean);
                    //获取到接口
                    Class<?>[] interfacesIfJdk = SpringProxyUtils.findInterfaces(bean);

                    //判断当前bean或者父类接口中是否有 @GlobalTransactional、@GlobalLock 注解
                    if (!existsAnnotation(new Class[]{serviceInterface})
                        && !existsAnnotation(interfacesIfJdk)) {
                        return bean;
                    }

                    if (globalTransactionalInterceptor == null) {
                        //创建一个全局的事务增强器
                        globalTransactionalInterceptor = new GlobalTransactionalInterceptor(failureHandlerHook);
                        ConfigurationCache.addConfigListener(
                                ConfigurationKeys.DISABLE_GLOBAL_TRANSACTION,
                                (ConfigurationChangeListener)globalTransactionalInterceptor);
                    }
                    interceptor = globalTransactionalInterceptor;
                }
                bean.getClass().getName(), beanName, interceptor.getClass().getName());
                //判断当前bean是否是代理对象，如果不是代理类就去调用 AbstractAutoProxyCreator.wrapIfNecessary() 对bean进行代理
                if (!AopUtils.isAopProxy(bean)) {
                    //其中又调用了 getAdvicesAndAdvisorsForBean() 方法，在当前类进行了覆写，其中直接返回了 interceptor 切面
                    bean = super.wrapIfNecessary(bean, beanName, cacheKey);
                } else {
                    //当前bean已经是代理类，通过反射获取到两种不同代理类的其中的属性值 AdvisedSupport （spring在对其bean代理时创建的代理会有一个属性 advised属性，这个字段里面存储了切面等数据；例如：JdkDynamicAopProxy）
                    AdvisedSupport advised = SpringProxyUtils.getAdvisedSupport(bean);
                    //将获取出来的 Advisor 进行包装
                    Advisor[] advisor = buildAdvisors(beanName, getAdvicesAndAdvisorsForBean(null, null, null));
                    int pos;
                    for (Advisor avr : advisor) {
                        pos = findAddSeataAdvisorPosition(advised, avr);
                        advised.addAdvisor(pos, avr);
                    }
                }
                //将bean设置到代理对象集当中，表明当前对象已经被代理了
                PROXYED_SET.add(beanName);
                return bean;
            }
        } catch (Exception exx) {
            throw new RuntimeException(exx);
        }
    }
```

### getAdvicesAndAdvisorsForBean()

覆写的父类方法，在 **AbstractAutoProxyCreator.wrapIfNecessary()** 中创建代理对象时进行调用，这里直接返回了 **interceptor** 类型，目前这里返回以下两种：

- GlobalTransactionalInterceptor：全局的事务增强器
- TccActionInterceptor：TCC模式的增强器

```java
protected Object[] getAdvicesAndAdvisorsForBean(Class beanClass, String beanName, TargetSource customTargetSource)
            throws BeansException {
    return new Object[]{interceptor};
}
```

### initClient()

初始化了以下两个远程调用客户端，用于发起请求和监听结果

- TmNettyRemotingClient：事务管理器客户端
- RmNettyRemotingClient：资源管理器客户端

```java
private void initClient() {
        if (StringUtils.isNullOrEmpty(applicationId) || StringUtils.isNullOrEmpty(txServiceGroup)) {
            throw new IllegalArgumentException(String.format("applicationId: %s, txServiceGroup: %s", applicationId, txServiceGroup));
        }
        //初始化TM客户端
        TMClient.init(applicationId, txServiceGroup, accessKey, secretKey);
        //初始化RM客户端
        RMClient.init(applicationId, txServiceGroup);
        if (LOGGER.isInfoEnabled()) {
            LOGGER.info("Resource Manager is initialized. applicationId[{}] txServiceGroup[{}]", applicationId, txServiceGroup);
        }
        //注册spring容器中的钩子函数
        registerSpringShutdownHook();

    }
```

## 3. GlobalTransactionalInterceptor

![](images/seata%E7%B1%BB%E5%9E%8B%E8%B0%83%E7%94%A8%E7%B1%BB.svg)

### invoke()

```java
public Object invoke(final MethodInvocation methodInvocation) throws Throwable {
        //目标源类的class对象
        Class<?> targetClass =
            methodInvocation.getThis() != null ? AopUtils.getTargetClass(methodInvocation.getThis()) : null;
        //当前执行的方法
        Method specificMethod = ClassUtils.getMostSpecificMethod(methodInvocation.getMethod(), targetClass);
        if (specificMethod != null && !specificMethod.getDeclaringClass().equals(Object.class)) {
            final Method method = BridgeMethodResolver.findBridgedMethod(specificMethod);
            //获取到当前执行方法的 @GlobalTransactional 注解
            final GlobalTransactional globalTransactionalAnnotation =
                getAnnotation(method, targetClass, GlobalTransactional.class);
            //获取到当前执行方法的 @GlobalLock 注解
            final GlobalLock globalLockAnnotation = getAnnotation(method, targetClass, GlobalLock.class);
            //判断是否关闭事务或者降低检查等级
            boolean localDisable = disable || (degradeCheck && degradeNum >= degradeCheckAllowTimes);
            if (!localDisable) {
                if (globalTransactionalAnnotation != null || this.aspectTransactional != null) {
                    AspectTransactional transactional;
                    //获取到注解中的数据转换成 AspectTransactional 对象
                    if (globalTransactionalAnnotation != null) {
                        transactional = new AspectTransactional(globalTransactionalAnnotation.timeoutMills(),
                            globalTransactionalAnnotation.name(), globalTransactionalAnnotation.rollbackFor(),
                            globalTransactionalAnnotation.noRollbackForClassName(),
                            globalTransactionalAnnotation.noRollbackFor(),
                            globalTransactionalAnnotation.noRollbackForClassName(),
                            globalTransactionalAnnotation.propagation(),
                            globalTransactionalAnnotation.lockRetryInterval(),
                            globalTransactionalAnnotation.lockRetryTimes());
                    } else {
                        transactional = this.aspectTransactional;
                    }
                    //处理全局事务
                    return handleGlobalTransaction(methodInvocation, transactional);
                } else if (globalLockAnnotation != null) {
                    //处理全局锁注解，会设置 KEY_GLOBAL_LOCK_FLAG 标识在 RootContext当中
                    return handleGlobalLock(methodInvocation, globalLockAnnotation);
                }
            }
        }
        //执行本体方法
        return methodInvocation.proceed();
    }
```

### handleGlobalTransaction()

当前方法中，通过 **transactionalTemplate.execute()** 执行事务，其中创建了一个 **TransactionalExecutor** 来进行处理之后的回调，在 **catch** 方法块中对异常进行捕获，对对应的异常code进行判断然后执行对应的操作

```java
Object handleGlobalTransaction(final MethodInvocation methodInvocation,
        final AspectTransactional aspectTransactional) throws Throwable {
        boolean succeed = true;
        try {
            //通过事务执行器，等待前置任务执行完成之后在执行回调函数中的方法
            return transactionalTemplate.execute(new TransactionalExecutor() {
                /**
                 * 直接执行本地方法
                 */
                @Override
                public Object execute() throws Throwable {
                    return methodInvocation.proceed();
                }

                /**
                 * 获取事务的名称
                 */
                public String name() {
                    String name = aspectTransactional.getName();
                    if (!StringUtils.isNullOrEmpty(name)) {
                        return name;
                    }
                    return formatMethod(methodInvocation.getMethod());
                }

                /**
                 * 获取到事务的信息
                 */
                @Override
                public TransactionInfo getTransactionInfo() {
                    // reset the value of timeout
                    int timeout = aspectTransactional.getTimeoutMills();
                    if (timeout <= 0 || timeout == DEFAULT_GLOBAL_TRANSACTION_TIMEOUT) {
                        timeout = defaultGlobalTransactionTimeout;
                    }

                    TransactionInfo transactionInfo = new TransactionInfo();
                    transactionInfo.setTimeOut(timeout);
                    transactionInfo.setName(name());
                    transactionInfo.setPropagation(aspectTransactional.getPropagation());
                    transactionInfo.setLockRetryInterval(aspectTransactional.getLockRetryInterval());
                    transactionInfo.setLockRetryTimes(aspectTransactional.getLockRetryTimes());
                    Set<RollbackRule> rollbackRules = new LinkedHashSet<>();
                    for (Class<?> rbRule : aspectTransactional.getRollbackFor()) {
                        rollbackRules.add(new RollbackRule(rbRule));
                    }
                    for (String rbRule : aspectTransactional.getRollbackForClassName()) {
                        rollbackRules.add(new RollbackRule(rbRule));
                    }
                    for (Class<?> rbRule : aspectTransactional.getNoRollbackFor()) {
                        rollbackRules.add(new NoRollbackRule(rbRule));
                    }
                    for (String rbRule : aspectTransactional.getNoRollbackForClassName()) {
                        rollbackRules.add(new NoRollbackRule(rbRule));
                    }
                    transactionInfo.setRollbackRules(rollbackRules);
                    return transactionInfo;
                }
            });
        } catch (TransactionalExecutor.ExecutionException e) {
            //根据对应的失败码，执行对应的操作
            TransactionalExecutor.Code code = e.getCode();
            switch (code) {
                case RollbackDone:
                    throw e.getOriginalException();
                case BeginFailure:
                    succeed = false;
                    failureHandler.onBeginFailure(e.getTransaction(), e.getCause());
                    throw e.getCause();
                case CommitFailure:
                    succeed = false;
                    failureHandler.onCommitFailure(e.getTransaction(), e.getCause());
                    throw e.getCause();
                case RollbackFailure:
                    failureHandler.onRollbackFailure(e.getTransaction(), e.getOriginalException());
                    throw e.getOriginalException();
                case RollbackRetrying:
                    failureHandler.onRollbackRetrying(e.getTransaction(), e.getOriginalException());
                    throw e.getOriginalException();
                default:
                    throw new ShouldNeverHappenException(String.format("Unknown TransactionalExecutor.Code: %s", code));
            }
        } finally {
            if (degradeCheck) {
                //发布一个时间消息
                EVENT_BUS.post(new DegradeCheckEvent(succeed));
            }
        }
    }
```

## 4. TransactionalTemplate

### execute()

根据设置定的事务传播机制来选择是否需要创建事务等操作，然后在执行本地业务方法前开启一个事务，事务的传播类型：

- NOT_SUPPORTED：不支持事务
- REQUIRES_NEW：存在事务，将原来的事务暂停，然后创建一个新的事务
- SUPPORTS：支持事务，不存在事务就直接执行
- REQUIRED：如果存在事务就以当前事务执行，不存在创建一个新的
- NEVER：不支持事务
- MANDATORY：如果不存在事务抛出异常，如果存在继续执行

```java
public Object execute(TransactionalExecutor business) throws Throwable {
        TransactionInfo txInfo = business.getTransactionInfo();
        if (txInfo == null) {
            throw new ShouldNeverHappenException("transactionInfo does not exist");
        }
        //获取到全局事务信息，RootContext.getXID() 全局事务id，如果不为空，创建一个 DefaultGlobalTransaction类型，角色是事务的参与者
        GlobalTransaction tx = GlobalTransactionContext.getCurrent();
        //事务的传播机制
        Propagation propagation = txInfo.getPropagation();
        SuspendedResourcesHolder suspendedResourcesHolder = null;
        try {
            //判断事务的传播类型
            switch (propagation) {
                case NOT_SUPPORTED:
                    //如果存在事务，需要将当前事务停止
                    if (existingTransaction(tx)) {
                        //解绑xid
                        suspendedResourcesHolder = tx.suspend();
                    }
                    //执行方法
                    return business.execute();
                case REQUIRES_NEW:
                    // 如果存在事务，将原来的事务暂停，然后创建一个新的事务
                    if (existingTransaction(tx)) {
                        suspendedResourcesHolder = tx.suspend();
                        //创建一个事务的发起者
                        tx = GlobalTransactionContext.createNew();
                    }
                    break;
                case SUPPORTS:
                    // 支持事务，如果不存在事务，直接执行方法；如果存在继续执行后续的方法
                    if (notExistingTransaction(tx)) {
                        return business.execute();
                    }
                    // Continue and execute with new transaction
                    break;
                case REQUIRED:
                    // 如果当前事务存在继续执行，如果不存在，下面会创建一个新的事务
                    break;
                case NEVER:
                    // 如果存在事务抛出异常
                    if (existingTransaction(tx)) {
                        throw new TransactionException(
                            String.format("Existing transaction found for transaction marked with propagation 'never', xid = %s"
                                    , tx.getXid()));
                    } else {
                        // 不存在事务直接执行
                        return business.execute();
                    }
                case MANDATORY:
                    // 如果事务不存在直接抛出异常
                    if (notExistingTransaction(tx)) {
                        throw new TransactionException("No existing transaction found for transaction marked with propagation 'mandatory'");
                    }
                    // 继续执行当前事务
                    break;
                default:
                    throw new TransactionException("Not Supported Propagation:" + propagation);
            }

            // 如果不存在事务，那么就创建一个事务发起者的新事务
            if (tx == null) {
                tx = GlobalTransactionContext.createNew();
            }

            // 创建一个全局锁的配置信息，这里返回的是被配置信息替换之前的信息
            GlobalLockConfig previousConfig = replaceGlobalLockConfig(txInfo);

            try {
                //开启一个事务，发送事务的请求到 TC（事务协调者，也就是Server端）
                beginTransaction(txInfo, tx);

                Object rs;
                try {
                    // 执行方法，在方法执行时，通过获取到数据源的代理类 DataSourceProxy，执行后续的sql解析创建前后镜像的逻辑
                    rs = business.execute();
                } catch (Throwable ex) {
                    //抛出异常之后需要进行操作
                    completeTransactionAfterThrowing(txInfo, tx, ex);
                    throw ex;
                }

                //没有异常就提交事务
                commitTransaction(tx);

                return rs;
            } finally {
                //最后清除全局锁的配置信息
                resumeGlobalLockConfig(previousConfig);
                //触发完成之后的钩子函数
                triggerAfterCompletion();
                //清除钩子函数
                cleanUp();
            }
        } finally {
            // 如果事务是暂停的状态，需要恢复它
            if (suspendedResourcesHolder != null) {
                tx.resume(suspendedResourcesHolder);
            }
        }
    }
```

### beginTransaction()

```java
private void beginTransaction(TransactionInfo txInfo, GlobalTransaction tx) throws TransactionalExecutor.ExecutionException {
    try {
        //触发开启事务之前的钩子函数
        triggerBeforeBegin();
        //开启事务，通过 TM 申请一个全局XID
        tx.begin(txInfo.getTimeOut(), txInfo.getName());
        //事务开启之后的钩子函数
        triggerAfterBegin();
    } catch (TransactionException txe) {
        //抛出一个 开始事务失败的异常信息，方便外层获取到了之后进行对应的操作
        throw new TransactionalExecutor.ExecutionException(tx, txe,
                                                           	 TransactionalExecutor.Code.BeginFailure);

    }
}
```

## 5. GlobalTransaction

全局事务接口，用于定义一些标准方法，通过子类 **DefaultGlobalTransaction** 实现，上面会调用 **begin()** 方法

```java
public interface GlobalTransaction {

    /**
     * 开启全局事务
     */
    void begin() throws TransactionException;
    void begin(int timeout) throws TransactionException;
    void begin(int timeout, String name) throws TransactionException;

    /**
     * 提交事务
     */
    void commit() throws TransactionException;

    /**
     * 回滚全局事务
     */
    void rollback() throws TransactionException;

    /**
     * 暂停全局事务
     */
    SuspendedResourcesHolder suspend() throws TransactionException;

    /**
     * 回复一个全局事务
     */
    void resume(SuspendedResourcesHolder suspendedResourcesHolder) throws TransactionException;

    /**
     * 查询当前事务的状态
     */
    GlobalStatus getStatus() throws TransactionException;

    /**
     * 获取XID
     */
    String getXid();

    /**
     * 上报全局事务的状态信息
     */
    void globalReport(GlobalStatus globalStatus) throws TransactionException;

    /**
     * 获取本地事务状态
     */
    GlobalStatus getLocalStatus();

    /**
     * 获取当前全局事务的角色，是事务参与者还是事务发起者
     */
    GlobalTransactionRole getGlobalTransactionRole();

}
```

### begin()

开启一个全局事务，会对当前事务的角色进行判断，是否应该存在 **XID**

```java
public void begin(int timeout, String name) throws TransactionException {
        //如果角色不为事务发起者的话，判断XID是否为null
        if (role != GlobalTransactionRole.Launcher) {
            assertXIDNotNull();
            if (LOGGER.isDebugEnabled()) {
                LOGGER.debug("Ignore Begin(): just involved in global transaction [{}]", xid);
            }
            return;
        }
        //如果角色为事务的发起者，那么XID需要为空
        assertXIDNull();
        String currentXid = RootContext.getXID();
        if (currentXid != null) {
            throw new IllegalStateException("Global transaction already exists," +
                " can't begin a new global transaction, currentXid = " + currentXid);
        }
        //通过TM 发起rpc请求获取到全局的XID
        xid = transactionManager.begin(null, null, name, timeout);
        //事务状态为开启
        status = GlobalStatus.Begin;
        RootContext.bind(xid);
        if (LOGGER.isInfoEnabled()) {
            LOGGER.info("Begin new global transaction [{}]", xid);
        }
    }
```

## 6. 一阶段

在一阶段的调用流程是

![](images/seata%20AT%E6%A8%A1%E5%BC%8F%E6%89%A7%E8%A1%8C%E6%B5%81%E7%A8%8B.svg)

### 6.1 DataSource

Seata最重要的一个功能就是对 **DataSource** 进行了代理，在用户执行插入 **sql** 时会在插入之间根据 **sql** 构建一个前置镜像出来，如果出现异常了，就可以通过 **undolog** 日志里面的镜像语句进行回滚；

seata中代理对数据进行代理的方式以及调用联调大致如下，seata对 **数据源、连接对象、预编译对象** 都进行了代理，最后通过 **ExecuteTemplate** 对象来执行解析 sql创建镜像等操作

![](images/seata%E6%95%B0%E6%8D%AE%E6%BA%90%E8%B0%83%E7%94%A8%E9%93%BE.svg)

### 6.2 SeataAutoDataSourceProxyCreator

步骤跟上面扫描 **@GloableTransactional** 一样对 **DataSource** 数据源进行代理，代理对象为 **SeataDataSourceProxy** 类型，根据代理模式创建不同的实现类：

- DataSourceProxy：AT模式
- DataSourceProxyXA：XA模式

```java
protected Object wrapIfNecessary(Object bean, String beanName, Object cacheKey) {
    // 只对 DataSource 进行代理
    if (!(bean instanceof DataSource)) {
        return bean;
    }
    if (!(bean instanceof SeataDataSourceProxy)) {
        //调用父类对 bean 进行代理
        Object enhancer = super.wrapIfNecessary(bean, beanName, cacheKey);
        //要么已经被代理了，要么是被排除了
        if (bean == enhancer) {
            return bean;
        }
        //否者构建代理对象
        DataSource origin = (DataSource) bean;
        SeataDataSourceProxy proxy = buildProxy(origin, dataSourceProxyMode);
        DataSourceProxyHolder.put(origin, proxy);
        return enhancer;
    }
    SeataDataSourceProxy proxy = (SeataDataSourceProxy) bean;
    DataSource origin = proxy.getTargetDataSource();
    Object originEnhancer = super.wrapIfNecessary(origin, beanName, cacheKey);
    if (origin == originEnhancer) {
        return origin;
    }
    DataSourceProxyHolder.put(origin, proxy);
    return originEnhancer;
}
```

### 6.3 SeataAutoDataSourceProxyAdvice

数据源通知类，通知类中没有做什么特别的事，就是判断了当前执行的方法在 **DataSource** 中是否也具有相同的方法，如果存在相同的方法，直接调用 **代理类** 就可以了

```java
public Object invoke(MethodInvocation invocation) throws Throwable {
    // check whether current context is expected
    if (!inExpectedContext()) {
        return invocation.proceed();
    }
    //获取需要执行的方法
    Method method = invocation.getMethod();
    String name = method.getName();
    Class<?>[] parameterTypes = method.getParameterTypes();
    //获取到DataSource中是否具有当前方法，如果抛出异常那么直接执行本体方法
    Method declared;
    try {
        declared = DataSource.class.getDeclaredMethod(name, parameterTypes);
    } catch (NoSuchMethodException e) {
        return invocation.proceed();
    }
    //调用代理对象的方法
    DataSource origin = (DataSource) invocation.getThis();
    SeataDataSourceProxy proxy = DataSourceProxyHolder.get(origin);
    Object[] args = invocation.getArguments();
    return declared.invoke(proxy, args);
}
```

### 6.4 SeataDataSourceProxy

```java
public interface SeataDataSourceProxy extends DataSource {

    /**
     * 获取到代理的源对象
     */
    DataSource getTargetDataSource();

    /**
     * 获取当前分支事务采用的模式
     */
    BranchType getBranchType();
}
```

![](images/DataSourceProxy.png)

### 6.5 DataSourceProxy

AT模式的数据源代理对象，通过 **DataSource** 的使用方式，一般是，先通过 **getConnection()** 获取到一个数据库的连接，然后通过 **Connection** 对象再创建一个 **Statement** 的对象来操作 **sql** 语句，所以这里我们可以先看 **getConnection()** 方法是如何实现的

#### getConnection()

直接创建了一个 **ConnectionProxy** 连接代理对象

```java
public ConnectionProxy getConnection() throws SQLException {
    Connection targetConnection = targetDataSource.getConnection();
    return new ConnectionProxy(this, targetConnection);
}
```

### 6.6 ConnectionProxy

![](images/ConnectionProxy.png)

#### createStatement()

又对 **Statement** 对象创建了一个代理对象

```java
public Statement createStatement() throws SQLException {
    Statement targetStatement = getTargetConnection().createStatement();
    return new StatementProxy(this, targetStatement);
}
```

#### commit()

##### 1）写隔离实现机制

提交事务时，会先向 **TC** 注册一个全局锁，表名+行记录 构成的锁；Seata 中对于写隔离的实现就是采用全局锁实现，写的过程：

- 一阶段本地事务提交前，申请全局锁，拿不到全局锁不能提交
- 超出了限制将放弃拿锁，回滚本地事务，释放本地锁

例如：t1和t2 两个事务，t1将 1000 修改为 900，t1先拿到全局锁，提交本地事务释放了本地锁，然后t2拿到本地锁将900修改为800，但是t1全局锁还没有释放，t2拿不到全局锁就根据策略进行重试，这时t1收到了 **TC** 的回滚请求，t1开始根据undolog进行回滚，发现t2还持有本地锁，就会一直进行重试回滚，t2持有本地锁重试次数超过了限制，放弃提交开始回滚数据释放问题锁，t1拿到本地锁开始执行回滚任务；因为整个过程 **全局锁** 在 t1结束前一直是被 t1 持有，所以不会发生脏写的问题。

##### 2）读隔离实现机制

在数据库本地事务隔离级别 **读已提交（Read Committed）** 或以上的基础上，Seata（AT 模式）的默认全局隔离级别是 **读未提交（Read Uncommitted）** 。

如果应用在特定场景下，必需要求全局的 **读已提交** ，目前 Seata 的方式是通过 SELECT FOR UPDATE 语句的代理。

为什么要默认采用 **读未提交**？

猜想：如果使用默认数据库的 **可重复读**，会出现的问题就是 t1开启事务修改1000 - 100 = 900但是本地事务还没有提交，t2查询的数据出来还是1000 - 100 = 900，t1拿到锁开始提交事务并且释放本地锁以及全局锁，而t2也开始提交事务，这时候 t1和t2 都将数据改成了900，就导致了数据异常

```java
public void commit() throws SQLException {
    try {
        //采用重试机制，反复的获取锁
        lockRetryPolicy.execute(() -> {
            //执行提交任务
            doCommit();
            return null;
        });
    } catch (SQLException e) {
        //判断是否是自动提交 并且并没有被改变过
        if (targetConnection != null && !getAutoCommit() && !getContext().isAutoCommitChanged()) {
            rollback();
        }
        throw e;
    } catch (Exception e) {
        throw new SQLException(e);
    }
}
```

#### doCommit()

```java
private void doCommit() throws SQLException {
    //查看是否存在全局事务的配置，如果存在全局事务需要注册当前分支
    if (context.inGlobalTransaction()) {
        //如果分支注册成功了，直接提交事务
        processGlobalTransactionCommit();
    } else if (context.isGlobalLockRequire()) {
        //如果不存在 XID 先去判断是否存在全局锁
        processLocalCommitWithGlobalLocks();
    } else {
        //提交事务
        targetConnection.commit();
    }
}
```

#### processGlobalTransactionCommit()

```java
private void processGlobalTransactionCommit() throws SQLException {
    try {
        //向 TC 设置一个由 表名:id 组成的全局锁，返回一个 branchId
        register();
    } catch (TransactionException e) {
        //如果锁已经存在，那么构建一个锁冲突的异常 LockConflictException
        recognizeLockKeyConflictException(e, context.buildLockKeys());
    }
    try {
        //刷新undolog日志，提交事务
        UndoLogManagerFactory.getUndoLogManager(this.getDbType()).flushUndoLogs(this);
        targetConnection.commit();
    } catch (Throwable ex) {
        LOGGER.error("process connectionProxy commit error: {}", ex.getMessage(), ex);
        //上报异常事务状态
        report(false);
        throw new SQLException(ex);
    }
    if (IS_REPORT_SUCCESS_ENABLE) {
        report(true);
    }
    context.reset();
}
```

### 6.7 PreparedStatementProxy

![](images/StatementProxy.png)

**PreparedStatementProxy** 覆写的 **execute()** 方法并没有做什么事，直接通过 **ExecuteTemplate** 来执行的

```java
@Override
public boolean execute() throws SQLException {
    //执行完之后，直接调用回调函数执行源sql
    return ExecuteTemplate.execute(this, (statement, args) -> statement.execute());
}
```

### 6.8 ExecuteTemplate

- SQLRecognizerFactory : sql识别工厂，引的哪个包就使用哪个识别器，这里使用的是 druid的识别器
  - DruidSQLRecognizerFactoryImpl
  - AntlrMySQLRecognizerFactory
- SQLRecognizer：sql识别器将对应的sql语句解析成不同的类型，下面是实现类，数据库不同实现的类型也不同

![1668495467504](images/1668495467504.png)

```java
public static <T, S extends Statement> T execute(List<SQLRecognizer> sqlRecognizers,
                                                     StatementProxy<S> statementProxy,
                                                     StatementCallback<T, S> statementCallback,
                                                     Object... args) throws SQLException {
        //对其进行全局锁的检查
        if (!RootContext.requireGlobalLock() && BranchType.AT != RootContext.getBranchType()) {
            //如果没有全局锁的标识并且也不是AT模式，直接执行源sql
            return statementCallback.execute(statementProxy.getTargetStatement(), args);
        }
        //创建sql识别器
        String dbType = statementProxy.getConnectionProxy().getDbType();
        if (CollectionUtils.isEmpty(sqlRecognizers)) {
            /**
             *  根据sql识别器工厂创建对应的识别器来创建 SQLRecognizer 对象，目前提供了两个实现类
             *  DruidSQLRecognizerFactoryImpl 和 AntlrMySQLRecognizerFactory 两个子类进行识别，两个子类通过两个包进行引用
             */
            sqlRecognizers = SQLVisitorFactory.get(
                    statementProxy.getTargetSQL(),
                    dbType);
        }
        Executor<T> executor;
        /**
         * 根据对应的sql类型创建出对应的执行器
         * 默认：PlainExecutor
         * 插入：InsertExecutor （会根据数据库的类型进行创建，这里只是指定接口名称）
         * 修改：UpdateExecutor
         * 删除：DeleteExecutor
         * select for update：SelectForUpdateExecutor
         * insert_on_duplicate_update：MySQLInsertOrUpdateExecutor
         * 多个sql执行：MultiExecutor
         */
        if (CollectionUtils.isEmpty(sqlRecognizers)) {
            executor = new PlainExecutor<>(statementProxy, statementCallback);
        } else {
            if (sqlRecognizers.size() == 1) {
                SQLRecognizer sqlRecognizer = sqlRecognizers.get(0);
                switch (sqlRecognizer.getSQLType()) {
                    case INSERT:
                        executor = EnhancedServiceLoader.load(InsertExecutor.class, dbType,
                                    new Class[]{StatementProxy.class, StatementCallback.class, SQLRecognizer.class},
                                    new Object[]{statementProxy, statementCallback, sqlRecognizer});
                        break;
                    case UPDATE:
                        executor = new UpdateExecutor<>(statementProxy, statementCallback, sqlRecognizer);
                        break;
                    case DELETE:
                        executor = new DeleteExecutor<>(statementProxy, statementCallback, sqlRecognizer);
                        break;
                    case SELECT_FOR_UPDATE:
                        executor = new SelectForUpdateExecutor<>(statementProxy, statementCallback, sqlRecognizer);
                        break;
                    case INSERT_ON_DUPLICATE_UPDATE:
                        switch (dbType) {
                            case JdbcConstants.MYSQL:
                            case JdbcConstants.MARIADB:
                                executor =
                                    new MySQLInsertOrUpdateExecutor(statementProxy, statementCallback, sqlRecognizer);
                                break;
                            default:
                                throw new NotSupportYetException(dbType + " not support to INSERT_ON_DUPLICATE_UPDATE");
                        }
                        break;
                    default:
                        executor = new PlainExecutor<>(statementProxy, statementCallback);
                        break;
                }
            } else {
                executor = new MultiExecutor<>(statementProxy, statementCallback, sqlRecognizers);
            }
        }
        T rs;
        try {
            /**
             * 执行时默认时执行 PlainExecutor中的方法
             * 其他类型执行 BaseTransactionalExecutor 中的方法
             */
            rs = executor.execute(args);
        } catch (Throwable ex) {
            if (!(ex instanceof SQLException)) {
                // Turn other exception into SQLException
                ex = new SQLException(ex);
            }
            throw (SQLException) ex;
        }
        return rs;
    }
```



### 6.9 Executor

根据sql识别器，识别出sql的类型是 insert、update还是delete等类型，根据不同的类型创建不同的sql执行器

![](images/Executor.png)

**ExecuteTemplate** 中对sql进行解析之后创建出不同类型的 **Eexcutor** 实现类，以 **Insert** sql为例子，解析出的类型是 **MysqlInsertOrUpdateExecutor** 该类型并没有实现 **execute(args)** 方法，往上找调用的方法在 **BaseTrasactionalExecutor** 类中

#### execute()

从上下文中获取了 **XID** 进行绑定

```java
public T execute(Object... args) throws Throwable {
    //获取到 XID
    String xid = RootContext.getXID();
    if (xid != null) {
        //如果XID不为空的话绑定到当前获取的连接代理对象中，证明当前连接已经绑定了全局事务，如果后续连接来绑定新全局事务抛出异常
        statementProxy.getConnectionProxy().bind(xid);
    }
    //判断是否需要全局锁，这里根据方法是否打了 @GlobalLock注解，GlobalTransactionalInterceptor中会根据 @GlobalLock 和 @GlobalTransactional注解进行不同的处理，如果是事务这里应该是false
    statementProxy.getConnectionProxy().setGlobalLockRequire(RootContext.requireGlobalLock());
    //AbstractDMLBaseExecutor
    return doExecute(args);
}
```

#### doExecute()

```java
public T doExecute(Object... args) throws Throwable {
    AbstractConnectionProxy connectionProxy = statementProxy.getConnectionProxy();
    //根据是否自动提交事务来进行区分执行的代码，如果是自动提交，那么需要把自动提交关闭，改成手动提交
    if (connectionProxy.getAutoCommit()) {
        return executeAutoCommitTrue(args);
    } else {
        //不需要提交留给二阶段通知提交时开始提交
        return executeAutoCommitFalse(args);
    }
}
```

#### executeAutoCommitFalse()

先根据sql语句构建出前置镜像 **TableRecords** ，然后执行sql语句，根据前置镜像的主键id构建出执行sql之后的查询sql语句，根据修改之后记录构建后置镜像

```java
public class TableRecords implements java.io.Serializable {
    private static final long serialVersionUID = 4441667803166771721L;
	//表的元数据信息
    private transient TableMeta tableMeta;
	//表名称
    private String tableName;
	//每一行数据对应的记录，里面又包括了每个字段的key和value
    private List<Row> rows = new ArrayList<Row>();
}
```

在后续提交时会根据构建的 **tableName:{主键值1},{主键值2}...{主键值N}** 记录锁对行加上锁，防止其他事务进行修改

```java
protected T executeAutoCommitFalse(Object[] args) throws Exception {
    if (!JdbcConstants.MYSQL.equalsIgnoreCase(getDbType()) && isMultiPk()) {
        throw new NotSupportYetException("multi pk only support mysql!");
    }
    //构建前置镜像
    TableRecords beforeImage = beforeImage();
    //执行sql语句
    T result = statementCallback.execute(statementProxy.getTargetStatement(), args);
    //获取到影响的行数
    int updateCount = statementProxy.getUpdateCount();
    if (updateCount > 0) {
        //构建后置镜像根据前置镜像的id查询出执行之后的sql记录
        TableRecords afterImage = afterImage(beforeImage);
        //构建undolog日志：构建锁，锁的结构是 表名:主键值1....{值键值n}，并且将锁添加到 lockKeysBuffer 在执行提交事务时会根据前面的锁对表数据上锁
        prepareUndoLog(beforeImage, afterImage);
    }
    return result;
}
```

#### executeAutoCommitTrue()

这里需要注意一点的是 **LockRetryPolicy** 对象，这里用到的是 **AbstractDMLBaseExecutor.LockRetryPolicy**

-  **AbstractDMLBaseExecutor.LockRetryPolicy**：继承至 **ConnectionProxy.LockRetryPolicy**，覆写了 **onException()**，抛出了异常会进行回滚
- **ConnectionProxy.LockRetryPolicy** :**onException()** 方法什么事都没有做

```java
protected T executeAutoCommitTrue(Object[] args) throws Throwable {
    ConnectionProxy connectionProxy = statementProxy.getConnectionProxy();
    try {
        //改变是否自动提交
        connectionProxy.changeAutoCommit();
        //锁重试的机制，如果一直拿不到锁就执行回滚的任务
        return new LockRetryPolicy(connectionProxy).execute(() -> {
            //构建前后镜像undolog日志
            T result = executeAutoCommitFalse(args);
            //执行事务的提交，看上面 ConnectionProxy.commit()
            connectionProxy.commit();
            return result;
        });
    } catch (Exception e) {
        // when exception occur in finally,this exception will lost, so just print it here
        LOGGER.error("execute executeAutoCommitTrue error:{}", e.getMessage(), e);
        //根据配置中 在锁冲突时的重试回滚机制，默认是true
        if (!LockRetryPolicy.isLockRetryPolicyBranchRollbackOnConflict()) {
            connectionProxy.getTargetConnection().rollback();
        }
        throw e;
    } finally {
        //清除事务相关的属性
        connectionProxy.getContext().reset();
        //自动提交打开
        connectionProxy.setAutoCommit(true);
    }
}
```



## 7. 网络请求

### 7.1 TransactionManager

事务管理器，在客户端主要用于发起事务请求、提交事务、回滚事务请求等，用于跟 TC 进行通信的类，其中获取当前接口的实现类是通过 **TransactionManagerHolder** 进行获取，然后通过 SPI 接口获取到默认实现类，这里的默认实现类是 **DefaultTransactionManager** 

```java
public interface TransactionManager {

    /**
     * 开启全局事务
     *
     * @param applicationId           应用id，指明当前事务所属的应用
     * @param transactionServiceGroup 事务服务的分组
     * @param name                    全局事务的名称
     * @param timeout                 全局事务的超时时间
     */
    String begin(String applicationId, String transactionServiceGroup, String name, int timeout)
        throws TransactionException;

    /**
     * 根据指定的xid进行事务的提交
     */
    GlobalStatus commit(String xid) throws TransactionException;

    /**
     * 根据xid进行事务回滚
     */
    GlobalStatus rollback(String xid) throws TransactionException;

    /**
     * 获取到当前事务的状态
     */
    GlobalStatus getStatus(String xid) throws TransactionException;

    /**
     * 上报对应的全局事务的状态
     */
    GlobalStatus globalReport(String xid, GlobalStatus globalStatus) throws TransactionException;
}
```

### 7.2 RMClient

RM客户端实例化工具类，用于创建 RM 客户端， **GlobalTransactionScanner.initClient()** 方法，在bean对象创建时进行初始化，与之相同的还有一个 **TMClient** 逻辑都一样

```java
public class RMClient {
    public static void init(String applicationId, String transactionServiceGroup) {
        //通过单例模式创建一个Netty的 RM客户端端
        RmNettyRemotingClient rmNettyRemotingClient = RmNettyRemotingClient.getInstance(applicationId, transactionServiceGroup);
        //设置默认的资源管理器
        rmNettyRemotingClient.setResourceManager(DefaultResourceManager.get());
        /**
         * 设置事务方法的处理器，默认 DefaultRMHandler，其中根据类型选择对于模式的处理器
         * RMHandlerAT
         * RMHandlerSaga
         * RMHandlerTCC
         * RMHandlerXA
         */
        rmNettyRemotingClient.setTransactionMessageHandler(DefaultRMHandler.get());
        //初始化时注册对于方法类型的处理器
        rmNettyRemotingClient.init();
    }
}
```

### 7.3 RemotingClient

远程调用客户端，用于 **Client** 与 **Server** 的通信，下面是依赖结构图，目前客户端的实现就以下两个：（TM用于一阶段的通信，RM用于二阶段接收TC对事务的处理）；

- RmNettyRemotingClient：资源管理器客户端，管理分支事务处理的资源，驱动分支事务提交或回滚，与TC进行通信以注册分支事务和报告分支事务的状态
- TmNettyRemotiongClient：事务管理器，定义全局事务的范围，开始全局事务、提交或回滚全局事务

![](images/RemotingClient.png)

#### 7.3.1 TmNettyRemotiongClient

采用了单例模式，底层使用 **Netty** 作为通讯框架，**TM** 主要用于发送请求然后处理发送请求过后的数据

```java
public static TmNettyRemotingClient getInstance() {
    if (instance == null) {
        synchronized (TmNettyRemotingClient.class) {
            if (instance == null) {
                NettyClientConfig nettyClientConfig = new NettyClientConfig();
                //配置消费的线程池
                final ThreadPoolExecutor messageExecutor = new ThreadPoolExecutor(
                    nettyClientConfig.getClientWorkerThreads(), nettyClientConfig.getClientWorkerThreads(),
                    KEEP_ALIVE_TIME, TimeUnit.SECONDS,
                    new LinkedBlockingQueue<>(MAX_QUEUE_SIZE),
                    new NamedThreadFactory(nettyClientConfig.getTmDispatchThreadPrefix(),
                                           nettyClientConfig.getClientWorkerThreads()),
                    RejectedPolicies.runsOldestTaskPolicy());
                //创建TM客户端
                instance = new TmNettyRemotingClient(nettyClientConfig, null, messageExecutor);
            }
        }
    }
    return instance;
}
```

##### init()

初始化方法，在上面 **GlobalTransactional.initClient()** 中进行调用，注册对应消息类型的处理器

```java
public void init() {
    // 注册消息类型的处理器
    registerProcessor();
    if (initialized.compareAndSet(false, true)) {
        super.init();
        if (io.seata.common.util.StringUtils.isNotBlank(transactionServiceGroup)) {
            getClientChannelManager().reconnect(transactionServiceGroup);
        }
    }
}
```

##### registerProcessor()

**TM** 注册的消息类型处理是以下几种：

- MessageType.TYPE_SEATA_MERGE_RESULT：合并请求的结果（）
- MessageType.TYPE_GLOBAL_BEGIN_RESULT：全局事务开启的结果处理（）
- MessageType.TYPE_GLOBAL_COMMIT_RESULT：提交事务的结果处理
- MessageType.TYPE_GLOBAL_REPORT_RESULT：重新上报的结果处理
- MessageType.TYPE_GLOBAL_ROLLBACK_RESULT：回滚的结果处理
- MessageType.TYPE_GLOBAL_STATUS_RESULT：状态的结果处理
- MessageType.TYPE_REG_CLT_RESULT
- MessageType.TYPE_BATCH_RESULT_MSG：批量发送的结果
- MessageType.TYPE_HEARTBEAT_MSG：心跳消息的处理

```java
private void registerProcessor() {
    // 1.registry TC response processor
    ClientOnResponseProcessor onResponseProcessor =
        new ClientOnResponseProcessor(mergeMsgMap, super.getFutures(), getTransactionMessageHandler());
    super.registerProcessor(MessageType.TYPE_SEATA_MERGE_RESULT, onResponseProcessor, null);
    ..................省略代码
}
```

#### 7.3.2 RmNettyRemotiongClient

主要用于处理 **TC** 端主动下发的数据监听，用于 **二阶段处理**，初始化方法跟 **TM** 一样，只是注册的处理器不一样

##### registerProcessor()

**RM** 管理主要用于处理来自服务端下发的请求，主要注册的类型有以下：

- MessageType.TYPE_BRANCH_COMMIT：分支提交（RmBranchCommitProcessor）
- MessageType.TYPE_BRANCH_ROLLBACK：分支回滚（RmBranchRollbackProcessor）
- MessageType.TYPE_RM_DELETE_UNDOLOG：删除undolog（RmUndoLogProcessor）
- MessageType.TYPE_SEATA_MERGE_RESULT：合并结果（ClientOnResponseProcessor）
- MessageType.TYPE_BRANCH_REGISTER_RESULT：分支注册结果处理
- MessageType.TYPE_BRANCH_STATUS_REPORT_RESULT：状态报告结果
- MessageType.TYPE_GLOBAL_LOCK_QUERY_RESULT：全局锁查询结果处理
- MessageType.TYPE_REG_RM_RESULT
- MessageType.TYPE_BATCH_RESULT_MSG：批量发送结果信息
- MessageType.TYPE_HEARTBEAT_MSG：心跳结果（ClientHeartbeatProcessor）

```java
private void registerProcessor() {
        // 1.registry rm client handle branch commit processor
        RmBranchCommitProcessor rmBranchCommitProcessor = new RmBranchCommitProcessor(getTransactionMessageHandler(), this);
        super.registerProcessor(MessageType.TYPE_BRANCH_COMMIT, rmBranchCommitProcessor, messageExecutor);
        // 2.registry rm client handle branch rollback processor
        RmBranchRollbackProcessor rmBranchRollbackProcessor = new RmBranchRollbackProcessor(getTransactionMessageHandler(), this);
        super.registerProcessor(MessageType.TYPE_BRANCH_ROLLBACK, rmBranchRollbackProcessor, messageExecutor);
        ...........省略相同代码
    }
```

**TM、RM** 都是采用 **netty** 进行通信，可以在两个对象的 **构造方法** 中看到设置的最终 **handler** 对象是 **io.seata.core.rpc.netty.AbstractNettyRemotingClient.ClientHandler** 对象；

**AbstractNettyRemoting.processMessage()** 中获取的又是根据消息类型区分的 **RemotingProcessor** 处理器，就是上面注册的处理器

```java
class ClientHandler extends ChannelDuplexHandler {
    @Override
    public void channelRead(final ChannelHandlerContext ctx, Object msg) throws Exception {
        if (!(msg instanceof RpcMessage)) {
            return;
        }
        //调用 io.seata.core.rpc.netty.AbstractNettyRemoting.processMessage 的方法
        processMessage(ctx, (RpcMessage) msg);
    }
}
```

#### 7.3.3 RemotingProcessor

- ClientHeartbeatProcessor：客户端心跳处理
- ClientOnResponseProcessor：客户端处理 TC 的回复消息
- ServerHeartbeatProcessor：服务端心跳处理
- ServerOnRequestProcessor：服务端用于处理客户端发送的分支注册、分支状态上报、全局事务开启、全局锁查询、全局事务回滚、全局事务状态的请求
- ServerOnResponseProcessor：服务端处理客户端发送的分支提交结果、分支回滚结果的处理器
- RmBranchCommitProcessor：分支提交的处理器
- RmBranchRollbackProcessor：分支回滚处理器
- RmUndoLogProcessor：undolog删除处理器

![1668743024863](images/1668743024863.png)

### 7.4 MessageTypeAware（请求实体）

请求实体的接口，就只定义了一个方法，用于请求的类型

```java
public interface MessageTypeAware {

    /**
     * 获取到请求类型的code码
     */
    short getTypeCode();

}
```

#### 7.4.1 AbstractTransactionRequestToTC

对于客户端来说，请求体就是 **TM --> TC** ，seata中服务端的实体处理结构就是下面这样，抽象出了一个 **AbstractTransactionRequestToTC** 的抽象类，（tm发起的请求实体，tc就需要转换成对应的实体处理）

![RmToTc](images/RmToTc.png)



```java
public abstract class AbstractTransactionRequestToTC extends AbstractTransactionRequest {

    /**
     * TC入栈处理器
     */
    protected TCInboundHandler handler;

    /**
     * 设置TC的入栈处理器
     *
     * @param handler the handler
     */
    public void setTCInboundHandler(TCInboundHandler handler) {
        this.handler = handler;
    }
}
```

#### 7.4.2 AbstractTransactionRequestToRM

对于TC来说，给RM发送请求的实体对象是 **AbstractTransactionRequestToRM**，那么 RM就需要转换成 **AbstractTransactionRequestToRM** 下面的对于类型

```java
public abstract class AbstractTransactionRequestToRM extends AbstractTransactionRequest {

    /**
     * 设置RM入栈处理器
     */
    protected RMInboundHandler handler;

    
    public void setRMInboundMessageHandler(RMInboundHandler handler) {
        this.handler = handler;
    }
}
```

#### 7.4.3 RMInboundHandler

```java
public interface RMInboundHandler {

    /**
     * 处理分支提交的请求
     */
    BranchCommitResponse handle(BranchCommitRequest request);

    /**
     * 处理分支回滚的请求
     */
    BranchRollbackResponse handle(BranchRollbackRequest request);

    /**
     * 处理undolog删除日志的请求
     */
    void handle(UndoLogDeleteRequest request);
}
```

#### 7.4.4 TCInboundHandler

```java
public interface TCInboundHandler {

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



上面的两个抽象类中，为什么要在实体类中定义两个 **handler** 呢？让我一起看看 seata 的实体设置，这样的用意是什么。例如下面这个 **分支提交请求** ，其中覆写了 **handle()** ，而这个 **hander** 的类型是上面两种 **TCInboundHandler、RMInboundHandler** 两个接口，两个接口给每一个请求方法都定义了一个方法，参数不同而已，也就是说，通过下面这种方法可以根据指定的参数调用到对应的处理方法，只需要设置定义的 **handler **即可

```java
public class BranchCommitRequest extends AbstractBranchEndRequest {

    @Override
    public short getTypeCode() {
        return MessageType.TYPE_BRANCH_COMMIT;
    }

    @Override
    public AbstractTransactionResponse handle(RpcContext rpcContext) {
        return handler.handle(this);
    }
}
```

**io.seata.rm.AbstractRMHandler#onRequest** ：这里的调用路径就是 netty 收到 TC 的分支提交请求，然后根据类型找到 **RmBranchCommitProcessor** 然后在 **process()** 中 调用 **DefaultRMHandler.onRequest()** 方法

```java
public AbstractResultMessage onRequest(AbstractMessage request, RpcContext context) {
        if (!(request instanceof AbstractTransactionRequestToRM)) {
            throw new IllegalArgumentException();
        }
        AbstractTransactionRequestToRM transactionRequest = (AbstractTransactionRequestToRM)request;
    	//客户端设置自己为handler，就可以根据对应的请求类型调用到指定的方法
        transactionRequest.setRMInboundMessageHandler(this);

        return transactionRequest.handle(context);
    }
```



## 8. 二阶段

### 8.1 分支提交

在二阶段 TC 向 TM 发起分支提交的请求时，通过 **BranchCommitRequest** 构建请求体，前面在网络请求里面提到了 seata 中是如何对请求体进行处理的，下面代码是 **处理器** 调用到 **AbstractRMHandler** 的代码；对应的处理器是 **RmBranchCommitProcessor**

```java
public AbstractResultMessage onRequest(AbstractMessage request, RpcContext context) {
        if (!(request instanceof AbstractTransactionRequestToRM)) {
            throw new IllegalArgumentException();
        }
        //设置当前类为handler处理器
        AbstractTransactionRequestToRM transactionRequest = (AbstractTransactionRequestToRM)request;
        transactionRequest.setRMInboundMessageHandler(this);
        //这里通过调用，会调用到 DefaultRMHandler.handle(BranchCommitRequest) 的方法中，然后再通过对应的模式来获取处理器执行
        return transactionRequest.handle(context);
    }
```

**AbstractCallback** 一个抽象的回调函数，实现了自定义的 **Callback**：

- onSuccess：成功之后回调
- onTransactionException：出现事务异常的回调
- onException：出现执行异常的回调

```java
public BranchCommitResponse handle(BranchCommitRequest request) {
        //处理分支提交的请求
        BranchCommitResponse response = new BranchCommitResponse();
        //自定义的回调抽象类 AbstractCallback，其中实现了异常捕获的方法
        exceptionHandleTemplate(new AbstractCallback<BranchCommitRequest, BranchCommitResponse>() {
            @Override
            public void execute(BranchCommitRequest request, BranchCommitResponse response)
                throws TransactionException {
                //执行分支提交
                doBranchCommit(request, response);
            }
        }, request, response);
        return response;
    }
```

执行分支提交的代码比较简单，获取到对应模式的资源管理器，然后调用对应的方法，默认是AT模式，这里就获取的是 **DataSourceManager** 类

```java
protected void doBranchCommit(BranchCommitRequest request, BranchCommitResponse response)
        throws TransactionException {
        //获取到对应的RM管理器，AT模式的执行器是 io.seata.rm.datasource.DataSourceManager.branchCommit() 开启一个异步的worker线程
        BranchStatus status = getResourceManager().branchCommit(request.getBranchType(), xid, branchId, resourceId,
            applicationData);
    .........
    }
```

可以看到下面源码中 **DataSourceManager.brancheCommit()** 方法就调用了一个异步工作线程，执行分支提交 ，里面将任务添加到一个队列中等待执行，执行的任务就是删除对应的的  **undolog** 逻辑比较简单，就不贴详细的源码了

```java
public BranchStatus branchCommit(BranchType branchType, String xid, long branchId, String resourceId,
                                     String applicationData) throws TransactionException {
        return asyncWorker.branchCommit(xid, branchId, resourceId);
    }
```

### 8.2 分支回滚

对应处理器 **RmBranchRollbackProcessor**，前面的处理逻辑都一样，之后调用的分支回滚方法不一样，上面分支提交是 **branchCommit()**，而这里是调用 **branchRollback()**

```java
public BranchStatus branchRollback(BranchType branchType, String xid, long branchId, String resourceId,
                                       String applicationData) throws TransactionException {
        //根据资源id，获取到对应的数据源代理
        DataSourceProxy dataSourceProxy = get(resourceId);
        if (dataSourceProxy == null) {
            throw new ShouldNeverHappenException(String.format("resource: %s not found",resourceId));
        }
        try {
            //根据db类型获取到undolog的管理器，这里的undo方法，调用的 io.seata.rm.datasource.undo.AbstractUndoLogManager.undo
            UndoLogManagerFactory.getUndoLogManager(dataSourceProxy.getDbType()).undo(dataSourceProxy, xid, branchId);
        } catch (TransactionException te) {
            //对应对应的异常码，设置对应的分支回滚状态
                new Object[]{branchType, xid, branchId, resourceId, applicationData, te.getMessage()});
            if (te.getCode() == TransactionExceptionCode.BranchRollbackFailed_Unretriable) {
                return BranchStatus.PhaseTwo_RollbackFailed_Unretryable;
            } else {
                return BranchStatus.PhaseTwo_RollbackFailed_Retryable;
            }
        }
        return BranchStatus.PhaseTwo_Rollbacked;
    }
```

io.seata.rm.datasource.undo.AbstractUndoLogManager#undo：

```java
public void undo(DataSourceProxy dataSourceProxy, String xid, long branchId) throws TransactionException {
        Connection conn = null;
        ResultSet rs = null;
        PreparedStatement selectPST = null;
        boolean originalAutoCommit = true;
        for (; ; ) {
            try {
                //获取到真实的连接对象
                conn = dataSourceProxy.getPlainConnection();
                // The entire undo process should run in a local transaction.
                if (originalAutoCommit = conn.getAutoCommit()) {
                    //修改自动提交为手动提交
                    conn.setAutoCommit(false);
                }
                // 查询undo log日志出来
                selectPST = conn.prepareStatement(SELECT_UNDO_LOG_SQL);
                //设置分支id
                selectPST.setLong(1, branchId);
                //xid
                selectPST.setString(2, xid);
                //执行查询语句
                rs = selectPST.executeQuery();
                boolean exists = false;
                //遍历查询出来的数据
                while (rs.next()) {
                    exists = true;
                    //如果服务端发送了多次回滚请求，这里只需要确保回滚了正确状态的日志
                    int state = rs.getInt(ClientTableColumnsName.UNDO_LOG_LOG_STATUS);
                    if (!canUndo(state)) {
                        if (LOGGER.isInfoEnabled()) {
                            LOGGER.info("xid {} branch {}, ignore {} undo_log", xid, branchId, state);
                        }
                        return;
                    }
                    //解析出context字段
                    String contextString = rs.getString(ClientTableColumnsName.UNDO_LOG_CONTEXT);
                    //解析出context为map
                    Map<String, String> context = parseContext(contextString);
                    //回去到rollback_info信息
                    byte[] rollbackInfo = getRollbackInfo(rs);
                    //获取到解析器名称，按照解析器进行序列化，默认使用jackson
                    String serializer = context == null ? null : context.get(UndoLogConstants.SERIALIZER_KEY);
                    UndoLogParser parser = serializer == null ? UndoLogParserFactory.getInstance()
                        : UndoLogParserFactory.getInstance(serializer);
                    //解析出回滚日志对象
                    BranchUndoLog branchUndoLog = parser.decode(rollbackInfo);

                    try {
                        // 设置当前线程采用的解析器的名称
                        setCurrentSerializer(parser.getName());
                        //获取到undolog 前后镜像数据信息
                        List<SQLUndoLog> sqlUndoLogs = branchUndoLog.getSqlUndoLogs();
                        if (sqlUndoLogs.size() > 1) {
                            Collections.reverse(sqlUndoLogs);
                        }
                        //遍历sql
                        for (SQLUndoLog sqlUndoLog : sqlUndoLogs) {
                            //获取到表的元数据信息
                            TableMeta tableMeta = TableMetaCacheFactory.getTableMetaCache(dataSourceProxy.getDbType()).getTableMeta(
                                conn, sqlUndoLog.getTableName(), dataSourceProxy.getResourceId());
                            sqlUndoLog.setTableMeta(tableMeta);
                            //根据数据库类型以及sql类型获取到对应的 undo执行器
                            AbstractUndoExecutor undoExecutor = UndoExecutorFactory.getUndoExecutor(
                                dataSourceProxy.getDbType(), sqlUndoLog);
                            undoExecutor.executeOn(conn);
                        }
                    } finally {
                        // 移除当前线程的序列化器
                        removeCurrentSerializer();
                    }
                }
                //如果存在undolog日志，需要删除undolog日志，跟业务代码一起提交事务，需要保证 undolog和日志回滚sql的一致性
                if (exists) {
                    //删除undolog日志
                    deleteUndoLog(xid, branchId, conn);
                    //执行事务提交
                    conn.commit();
                } else {
                    //如果不存在undolog日志，说明视图提交出现了异常导致undolog日志没有被存储上，这里插入了一个 GlobalFinished 状态的日志防止视图被正确的提交了
                    insertUndoLogWithGlobalFinished(xid, branchId, UndoLogParserFactory.getInstance(), conn);
                    //执行事务提交
                    conn.commit();
                }
                return;
            } catch (SQLIntegrityConstraintViolationException e) {
                
            } catch (Throwable e) {
                //抛出了异常就执行undolog日志回滚
            } finally {
                //关闭各个流和管道
            }
        }
    }
```

方法中真正执行日志回滚的是 **AbstractUndoExecutor** 类型，这是一个抽象类，通过工厂方法进行创建，根据**数据库**类型以及 **sql的类型** 获取到对应的执行器

![](images/AbstractUndoExecutor.png)

io.seata.rm.datasource.undo.AbstractUndoExecutor#executeOn：

**executeOn** 作为一个抽象类的公关方法，其中一些比较有特点的函数方法都交给子类实现，例如：buildUndoSQL（构建回滚的sql，如果是插入构建的回滚sql就是删除）、getUndoRows（获取到后置镜像的数据信息）

```java
public void executeOn(Connection conn) throws SQLException {
        /**
         * 是否开启镜像数据的验证，如果开启了需要对后置镜像和当前数据进行数据的对比，如果不相同说明被其他事务改变需要根据对应的策略进行处理
         * 对比策略：
         *      前后镜像相比较是否一样，一样的就不需要执行后续了
         *      后镜与当前数据进行比较是否一样
         *          如果一样直接后续回滚
         *          如果不一样再判断前镜与当前数据是否一样，如果不一样说明出现了脏数据，如果一样就不需要执行后续了
         */
        if (IS_UNDO_DATA_VALIDATION_ENABLE && !dataValidationAndGoOn(conn)) {
            return;
        }
        PreparedStatement undoPST = null;
        try {
            //buildUndoSQL() 抽象方法用于子类进行实现，sql不同子类实现方法也不同；例如：insert，构建回滚函数就是delete函数删除对应的语句
            String undoSQL = buildUndoSQL();
            //根据前置镜像的sql构建出sql语句
            undoPST = conn.prepareStatement(undoSQL);
            //根据后置镜像的记录获取到对应需要的数据
            TableRecords undoRows = getUndoRows();
            for (Row undoRow : undoRows.getRows()) {
                ArrayList<Field> undoValues = new ArrayList<>();
                //获取到记录的主键值
                List<Field> pkValueList = getOrderedPkList(undoRows, undoRow, getDbType(conn));
                for (Field field : undoRow.getFields()) {
                    if (field.getKeyType() != KeyType.PRIMARY_KEY) {
                        undoValues.add(field);
                    }
                }
                //设置value
                undoPrepare(undoPST, undoValues, pkValueList);
                //执行sql语句
                undoPST.executeUpdate();
            }

        } 

    }
```

### 8.3 undolog删除

undolog日志的删除就比较简单了，调用处理 **io.seata.rm.RMHandlerAT#handle** 方法，通过undolog管理器删除执行，由 TC 指定删除对应的期限的 undolog日志，默认是删除过去 7 天的 3000条数据

io.seata.rm.RMHandlerAT#handle

```java
public void handle(UndoLogDeleteRequest request) {
        String resourceId = request.getResourceId();
        DataSourceManager dataSourceManager = (DataSourceManager)getResourceManager();
        DataSourceProxy dataSourceProxy = dataSourceManager.get(resourceId);
        boolean hasUndoLogTable = undoLogTableExistRecord.computeIfAbsent(resourceId, id -> checkUndoLogTableExist(dataSourceProxy));
        //根据存储天数进行数据的删除
        Date division = getLogCreated(request.getSaveDays());
        //获取到undolog日志管理器
        UndoLogManager manager = getUndoLogManager(dataSourceProxy);
        try (Connection conn = getConnection(dataSourceProxy)) {
            if (conn == null) {
                LOGGER.warn("Failed to get connection to delete expired undo_log for {}", resourceId);
                return;
            }
            int deleteRows;
            do {
                //根据日期进行删除，默认 7 天之内的 3000条数据
                deleteRows = deleteUndoLog(manager, conn, division);
            } while (deleteRows == LIMIT_ROWS);
        } catch (Exception e) {
            // should never happen, deleteUndoLog method had catch all Exception
        }
    }
```

