---
title: Seata实战入门与实战
date: 2024-12-30 19:07:05
updated: 2024-12-30 19:07:05
tags:
  - 分布式
  - Seata
comments: true
categories:
  - 分布式
  - Seata
thumbnail: https://images.unsplash.com/photo-1581091013158-5c7184f43b62?crop=entropy&cs=srgb&fm=jpg&ixid=M3w2NDU1OTF8MHwxfHJhbmRvbXx8fHx8fHx8fDE3NDMwODM0MTR8&ixlib=rb-4.0.3&q=85&w=1920&h=1080
---

# 分布式事务

## 一、分布式事务的组成部分

- 事务参与者：对应的一个一个的微服务
- 资源服务器：对应一个个微服务的数据库
- 事务管理器：决策各个事务参与者的提交和回滚

### 两阶段提交：

1. 准备阶段：向事务管理器向事务参与者发送预备请求，事务参与者在写本地的redo和undo日志，但是不提交，并且返回准备就绪的信息，最后提交的动作交给第二阶段来进行
2. 提交阶段：如果事务协调者收到失败或者超时的信息，直接给每个参与者发送回滚消息；否则提交消息，最后根据协调者的指令释放所有事务处理过程中使用的资源锁



## 二、项目例子

当前依赖，全局事务XID，不需要手动进行绑定，自动进行传递

```java
<dependency>
            <groupId>com.alibaba.cloud</groupId>
            &lt;!&ndash;加入spring-cloud-alibaba-seata，解决xid不传递问题&ndash;&gt;
            <artifactId>spring-cloud-alibaba-seata</artifactId>
            <version>2.2.0.RELEASE</version>
            <exclusions>
                <exclusion>
                    <groupId>io.seata</groupId>
                    <artifactId>seata-spring-boot-starter</artifactId>
                </exclusion>
                <exclusion>
                    <groupId>io.seata</groupId>
                    <artifactId>seata-all</artifactId>
                </exclusion>
            </exclusions>
</dependency>

```

全局事务XID需要通过过滤器或拦截器进行手动绑定，否则下游服务获取不到全局XID回滚不了

```java
<dependency>
            <groupId>io.seata</groupId>
            <artifactId>seata-spring-boot-starter</artifactId>
            <version>1.3.0</version>
            <!-- 这里需要排除自身的seata-all -->
            <exclusions>
                <exclusion>
                    <artifactId>seata-all</artifactId>
                    <groupId>io.seata</groupId>
                </exclusion>
            </exclusions>
        </dependency>
        <!-- 导入与之前下载的seata版本一致的包 -->
        <dependency>
            <groupId>io.seata</groupId>
            <artifactId>seata-all</artifactId>
            <version>1.3.0</version>
        </dependency>
```

OpenFeign进行手动传递XID

```java
@Component
public class FeignConfiguration implements RequestInterceptor {

    @Override
    public void apply(RequestTemplate requestTemplate) {
        requestTemplate.header("XID", RootContext.getXID());
    }
}
```

提供者手动绑定XID

```java
@Component
public class SeataFilter implements Filter {
    @Override
    public void doFilter(ServletRequest servletRequest, ServletResponse servletResponse, FilterChain filterChain) throws IOException, ServletException {
        HttpServletRequest request = (HttpServletRequest) servletRequest;
        //手动绑定XID
        String xid = request.getHeader("XID");
        if(StringUtils.isNotBlank(xid)){
            RootContext.bind(xid);
        }
        filterChain.doFilter(servletRequest,servletResponse);
    }
}
```

**seata在1.0.0版本之后就不需要手动进行数据源代理，已经被自动代理**

客户端的配置文件

```yml
seata:
  enabled: true
  tx-service-group: my_first_seata  #配置文件中的事务服务组一样
  config:
    type: nacos                 # nacos中拉去对应的配置文件
    nacos:
      server-addr: 192.168.60.46:8849
      group: SEATA_GROUP
  registry:  # 会去nacos中拉去seata-server服务
    type: nacos
    nacos:
      application: seata-server
      server-addr: 192.168.60.46:8849
      group: SEATA_GROUP
```

**seata1.0.0之后config文件下就移除了nacos-config.txt等文件，改为了config.txt需要手动下载，并且config.txt需要在nacos-config.sh的上一级目录下才能推送到nacos中**

```application
# 只需要修改下面几种配置即可，这里是配置客户端需要拉取的配置文件
service.vgroupMapping.自定义的名称=default
store.mode=db #修改为db
store.db.dbType=mysql #修改msql的连接方式账号和密码
```



## 三、seata原理

### 1、角色划分

- RM：资源管理者/事务参与者，也可以是一个TM

- TM：事务管理者，也是一个微服务，充当分布式事务的发起者

- TC：全局事务协调者seata-server，一个包需要搭建，TC来决定事务的回滚和提交

### 2、AT模式

#### （1）核心概念

- 两阶段提交：只执行，不提交

- seata 核心概念：边执行，边提交（两阶段的变种）
  - 一阶段：查询前置快照---------->执行业务语句-------------->查询出后置快照，保存只undo_log日志表中
  - 二阶段提交：分支插入待删除队列--------->异步删除undo_log表中数据
  - 二阶段回滚：根据配置选项选择是否检验dirty data------------>构造方向SQL----------->删除undo_log

#### （2）执行流程

阶段一：业务SQL：update product set name = 'GTS' where name = 'TXC'

- 解析SQL，根据update product解析出update语句，表product，条件where等相关信息
- 查询前置镜像：根据解析sql生成查询语句：**select id**, **name**, since **from** product **where name** = 'TXC'
- 执行业务SQL：update product set name = 'GTS' where name = 'TXC' 更新数据
- 查询后置镜像：通过主键定位数据
- 插入回滚日志：把前后镜像数据以及业务SQL相关的信息组成一条回滚日志记录，插入到undo_log表中
- 提交前，向TC注册分支，申请product表中，主键值记录的全局锁
- 本地事务提交：业务数据的更新和前面步骤中生成的undo_log一并提交
- 将本地事务的提交结果上报给TC

**业务数据和回滚日志记录会在同一个本地事务中保存，会释放本地锁和连接资源**

阶段二（回滚）：

- 收到TC的分支回滚请求，开启一个本地事务，把请求放入一个异步任务的队列里面
- 根据XID和Branch ID查找到相应的undo_log记录
- 数据校验：拿undo_log中的后镜与当前数据进行比较，如果有不同，说明当前数据被其它事务所更改，需要通过配置的策略进行处理
- 根据undo_log的前置镜像和业务sql的相关信息组成回滚语句
- 将分支回滚的结果提交给TC

通过一阶段的回滚日志进行反向补偿

阶段二（提交）：

- 收到TC的分支提交请求，把请求放入异步队列中，马上返回提交成功的结果给TC
- 异步批量的删除undo_log记录

#### （3）写隔离

**一阶段提交本地事务，必须需要拿到更改数据的全局锁，拿不到全局锁，不能提交本地事务，超出等待时间，会回滚本地事务，释放本地锁**

例：tx1和tx2两个全局事务同时修改 a表的m字段，m初始为1000；

tx1先开始，拿到本地锁，将m 1000-100 = 900。本地事务提交前，先拿到该记录的全局锁，本地提交释放本地锁。tx2开始，拿到本地锁，将m 900-100=800，提交本地事务前，先获取该记录的全局锁，tx1全局事务提交前，全局锁会被tx1所持有，tx2就会重试等待全局锁。

**tx1二阶段全局提交，释放全局锁。tx2拿到全局锁提交本地事务。如果tx1二阶段为全局回滚，那么会重新重试获取本地锁，此时tx2如果还在等待全局锁，同时持有本地锁，tx1分支事务就会等待tx2超时释放本地锁之后，再次获取本地锁；整个过程 全局锁都是被 tx1锁持有，不会存在脏数据的问题**

#### （4）读隔离

Seata AT模式的默认全局隔离级别是读未提交，如果在特定场景下，必需要求全局的读已提交，Seata采用通过select for update 语句来进行代理的；select for update语句的执行会申请*全局锁* ，如果全局锁被其它事务锁持有，就会回滚select for update的本地执行并且重试，因为这时候查询是被锁住，直到全局锁拿到，即读取相关的数据是已提交的

### 3、TCC模式

AT模式是基于本地支持ACID事务的关系型数据库：

- 一阶段prepare行为：在本地事务中，一并提交数据更新和相应的回滚记录
- 二阶段commit行为：马上成功，自动异步删除回滚记录
- 二阶段rollback行为：通过回滚日志，自动生成补偿操作，完成数据回滚

相应的TCC模式，不依赖本地底层数据的事务支持：

- 一阶段prepare行为：调用自定义的prepare逻辑
- 二阶段commit行为：调用自定义的commit逻辑
- 二阶段rollback行为：调用自定义的rollback逻辑

### 4、Saga模式

- 特点：业务流程中每个参与者都提交本地事务,当某一个参与者失败则补偿前面已经成功的参与者，一阶段正向服务和二阶段补偿服务都由业务开发者实现
- sage实现：基于状态机引擎来实现
  - 通过状态图来定义服务调用的流程并生成json状态语言定义文件
  - 状态图中一个节点可以是调用一个服务，节点可以配置它的补偿节点
  - 状态图json由状态机引擎驱动执行，当出现异常时状态引擎反向执行已经成功节点对应的补偿节点将事务回滚（用户可以自定义是否进行补偿）
  - 可以实现服务编排需求，支持单项选择、并发、子流程、参数转换、参数映射、服务执行状态判断、异常捕获等功能

### 5、XA模式

- 特点：利用事务资源（数据库、消息服务等）对 XA 协议的支持，以 XA 协议的机制来管理分支事务的一种 事务模式

