---
title: SpringCloud之Eureka Client源码解析
date: 2024-12-30 19:07:05
updated: 2024-12-30 19:07:05
tags:
  - Java
  - 源码
comments: true
categories:
  - Java
  - 源码
  - Spring
  - SpringCloud
thumbnail: https://images.unsplash.com/photo-1581091013158-5c7184f43b62?crop=entropy&cs=srgb&fm=jpg&ixid=M3w2NDU1OTF8MHwxfHJhbmRvbXx8fHx8fHx8fDE3NDMxNTE3ODR8&ixlib=rb-4.0.3&q=85&w=1920&h=1080
published: true
---

# 分析 Eureka之Client 源码

maven包，下面是引入 **Spring Cloud Eureka Client** 的依赖包这里 **Eureka Client** 采用的版本是 **2.2.5.RELEASE**

```maven
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-starter-netflix-eureka-client</artifactId>
</dependency>
```

下面的几个包用于注册 **eureka** 客户端以及 **ribbon** 和 **eureka** 进行服务关联的自动装配

- spring-cloud-commons：spring cloud定义的一些顶层接口，用于提供服务注册的一些能力
- spring-cloud-netflix-eureka-client：eureka的一些自动装配
- spring-cloud-netflix-ribbon：用于eureka跟RestTemplate进行负载均衡关联的自动装配



## 1. EurekaClientAutoConfiguration

**Eureka** 客户端的自动装配类

```java
@Configuration(proxyBeanMethods = false)
//开启配置
@EnableConfigurationProperties
//判断classpath下有没有EurekaClientConfig.class
@ConditionalOnClass(EurekaClientConfig.class)
//配置文件中是否有当前属性，默认为true
@ConditionalOnProperty(value = "eureka.client.enabled", matchIfMissing = true)
//判断配置文件属性中是否有 spring.cloud.discovery.enabled 属性，默认为true
@ConditionalOnDiscoveryEnabled
//在下面三个自动配置之前执行
@AutoConfigureBefore({ NoopDiscoveryClientAutoConfiguration.class,
		CommonsClientAutoConfiguration.class, ServiceRegistryAutoConfiguration.class })
//在下面自动装配之后执行
@AutoConfigureAfter(name = {
		"org.springframework.cloud.netflix.eureka.config.DiscoveryClientOptionalArgsConfiguration",
		"org.springframework.cloud.autoconfigure.RefreshAutoConfiguration",
		"org.springframework.cloud.netflix.eureka.EurekaDiscoveryClientConfiguration",
		"org.springframework.cloud.client.serviceregistry.AutoServiceRegistrationAutoConfiguration" })
public class EurekaClientAutoConfiguration { }
```

下面自配配置需要在 **EurekaClientAutoConfiguration** 自动装配之前执行

- DiscoveryClientOptionalArgsConfiguration：设置了 tls 配置加密证书，发送http请求进行加密
- RefreshAutoConfiguration：自动配置刷新
- EurekaDiscoveryClientConfiguration：开启客户端发现
- AutoServiceRegistrationAutoConfiguration：自动服务注册

下面自配配置需要在 **EurekaClientAutoConfiguration** 自动装配之后执行

- NoopDiscoveryClientAutoConfiguration
- CommonsClientAutoConfiguration
- ServiceRegistryAutoConfiguration：主要是通过注册接口用于操作服务在注册中心的状态



### 1.1 EurekaInstanceConfigBean

**Eureka** 配置bean对象，通过读取到对应的配置文件属性，进行创建当前客户端的实例信息

```java
@Bean
	@ConditionalOnMissingBean(value = EurekaInstanceConfig.class,
			search = SearchStrategy.CURRENT)
	public EurekaInstanceConfigBean eurekaInstanceConfigBean(InetUtils inetUtils,
			ManagementMetadataProvider managementMetadataProvider) {
        // DefaultManagementMetadataProvider默认创建的元数据提供器
		String hostname = getProperty("eureka.instance.hostname");
		boolean preferIpAddress = Boolean
				.parseBoolean(getProperty("eureka.instance.prefer-ip-address"));
		String ipAddress = getProperty("eureka.instance.ip-address");
		boolean isSecurePortEnabled = Boolean
				.parseBoolean(getProperty("eureka.instance.secure-port-enabled"));

		String serverContextPath = env.getProperty("server.servlet.context-path", "/");
		int serverPort = Integer.parseInt(
				env.getProperty("server.port", env.getProperty("port", "8080")));

		Integer managementPort = env.getProperty("management.server.port", Integer.class);
		String managementContextPath = env
				.getProperty("management.server.servlet.context-path");
		Integer jmxPort = env.getProperty("com.sun.management.jmxremote.port",
				Integer.class);
		EurekaInstanceConfigBean instance = new EurekaInstanceConfigBean(inetUtils);

		instance.setNonSecurePort(serverPort); //设置服务不安全的端口
		instance.setInstanceId(getDefaultInstanceId(env)); //设置默认的InstanceId
		instance.setPreferIpAddress(preferIpAddress);//是否优先使用服务的IP地址
		instance.setSecurePortEnabled(isSecurePortEnabled);//是否开启安全端口
		if (StringUtils.hasText(ipAddress)) {
            //设置ip地址
			instance.setIpAddress(ipAddress);
		}

		if (isSecurePortEnabled) {
            //如果开启了安全端口，设置安全端口
			instance.setSecurePort(serverPort);
		}

		if (StringUtils.hasText(hostname)) {
            //设置hostname
			instance.setHostname(hostname);
		}
		String statusPageUrlPath = getProperty("eureka.instance.status-page-url-path");
		String healthCheckUrlPath = getProperty("eureka.instance.health-check-url-path");

		if (StringUtils.hasText(statusPageUrlPath)) {
            //设置状态页的url地址
			instance.setStatusPageUrlPath(statusPageUrlPath);
		}
		if (StringUtils.hasText(healthCheckUrlPath)) {
            //设置健康检查的url路径
			instance.setHealthCheckUrlPath(healthCheckUrlPath);
		}
		//元数据管理器，优先使用元数据管理中设置的状态地址和健康检查地址
		ManagementMetadata metadata = managementMetadataProvider.get(instance, serverPort,
				serverContextPath, managementContextPath, managementPort);

		if (metadata != null) {
            //http://localhost:9003/actuator/health、http://localhost:9003/actuator/info 默认地址
			instance.setStatusPageUrl(metadata.getStatusPageUrl());
			instance.setHealthCheckUrl(metadata.getHealthCheckUrl());
			if (instance.isSecurePortEnabled()) {
				instance.setSecureHealthCheckUrl(metadata.getSecureHealthCheckUrl());
			}
			Map<String, String> metadataMap = instance.getMetadataMap();
			metadataMap.computeIfAbsent("management.port",
					k -> String.valueOf(metadata.getManagementPort()));
		}
		else {
            //拼接上管理上下文路径 /
			if (StringUtils.hasText(managementContextPath)) {
				instance.setHealthCheckUrlPath(
						managementContextPath + instance.getHealthCheckUrlPath());
				instance.setStatusPageUrlPath(
						managementContextPath + instance.getStatusPageUrlPath());
			}
		}
```

### 1.2 AbstractDiscoveryClientOptionalArgs

服务发现的参数配置类，在 **DiscoveryClientOptionalArgsConfiguration** 配置类中进行装配，会进行 **TLS** 的配置，在服务进行注册时，会使用当前 **MutableDiscoveryClientOptionalArgs** 中的 **健康、回调、预注册处理器**

```java
@Bean
@ConditionalOnClass(name = "com.sun.jersey.api.client.filter.ClientFilter")
@ConditionalOnMissingBean(value = AbstractDiscoveryClientOptionalArgs.class,
                          search = SearchStrategy.CURRENT)
public MutableDiscoveryClientOptionalArgs discoveryClientOptionalArgs(
    TlsProperties tlsProperties) throws GeneralSecurityException, IOException {
    logger.info("Eureka HTTP Client uses Jersey");
    MutableDiscoveryClientOptionalArgs result = new MutableDiscoveryClientOptionalArgs();
    setupTLS(result, tlsProperties);
    return result;
}
```



### 1.3 EurekaClient

下面是 **EurekaClient** 的依赖结构图，

![1657617436375](https://cdn.jsdelivr.net/gh/haijun823/note-picture@main/note-picture1657617436375.png)

```java
@Bean(destroyMethod = "shutdown")
@ConditionalOnMissingBean(value = EurekaClient.class,
                          search = SearchStrategy.CURRENT)
public EurekaClient eurekaClient(ApplicationInfoManager manager,//应用信息管理器，保存Eureka相关的状态
                                 EurekaClientConfig config) {
    // optionArgs 在 DiscoveryClientOptionalArgsConfiguration 中创建的MutableDiscoveryClientOptionalArgs对象，其中会进行SSL的配置
    return new CloudEurekaClient(manager, config, this.optionalArgs,
                                 this.context);
}
```

**CloudEurekaClient** 构造方法中调用了父类 **DiscoverClient** 父类构造方法

```java
public CloudEurekaClient(ApplicationInfoManager applicationInfoManager,
			EurekaClientConfig config, AbstractDiscoveryClientOptionalArgs<?> args,
			ApplicationEventPublisher publisher) {
    super(applicationInfoManager, config, args); //调用父类 DiscoverClient 中的构造方法
    this.applicationInfoManager = applicationInfoManager;
    this.publisher = publisher;
    this.eurekaTransportField = ReflectionUtils.findField(DiscoveryClient.class,
                                                          "eurekaTransport");
    ReflectionUtils.makeAccessible(this.eurekaTransportField);
}
```

下面是 **DiscoverClient** 构造函数中的代码，比较长，一点一点的看；其中使用了 **AbstractDiscoveryClientOptionalArgs** 里面进行配置的 **健康、回调、预注册处理器** 的 **Provider** 进行处理

```java
DiscoveryClient(ApplicationInfoManager applicationInfoManager, EurekaClientConfig config, AbstractDiscoveryClientOptionalArgs args,
                    Provider<BackupRegistry> backupRegistryProvider, EndpointRandomizer endpointRandomizer) {
    // 这里会通过上面注入的配置参数类，获取到其中对应的检查检查处理器以及回调函数处理器
        if (args != null) {
            this.healthCheckHandlerProvider = args.healthCheckHandlerProvider;
            this.healthCheckCallbackProvider = args.healthCheckCallbackProvider;
            this.eventListeners.addAll(args.getEventListeners());//获取到其中配置的监听器
            this.preRegistrationHandler = args.preRegistrationHandler;
        } else {
            this.healthCheckCallbackProvider = null;
            this.healthCheckHandlerProvider = null;
            this.preRegistrationHandler = null;
        }
        
        this.applicationInfoManager = applicationInfoManager;
    //获取到当前客户端的实例信息，默认启动状态为 STARTING 目的是为了，在创建时将当前客户端进行注册，注册成功，但是状态为 STARING 时也不能进行服务的调用，只有后续的 EurekaAutoServiceRegistration 中当spring启动完成之后才会进行状态的修改
        InstanceInfo myInfo = applicationInfoManager.getInfo();

        clientConfig = config;
        staticClientConfig = clientConfig;
        transportConfig = config.getTransportConfig();
        instanceInfo = myInfo;
        if (myInfo != null) {
            //设置app的路径表示，通过名称和id组成
            appPathIdentifier = instanceInfo.getAppName() + "/" + instanceInfo.getId();
        } else {
            logger.warn("Setting instanceInfo to a passed in null value");
        }

        this.backupRegistryProvider = backupRegistryProvider;
        this.endpointRandomizer = endpointRandomizer;
        this.urlRandomizer = new EndpointUtils.InstanceInfoBasedUrlRandomizer(instanceInfo);
        localRegionApps.set(new Applications());//初始化服务注册表

        fetchRegistryGeneration = new AtomicLong(0);

    //设置拉取注册表的对应区域的服务信息，使用逗号分割开
        remoteRegionsToFetch = new AtomicReference<String>(clientConfig.fetchRegistryForRemoteRegions());
    //分割区域信息
        remoteRegionsRef = new AtomicReference<>(remoteRegionsToFetch.get() == null ? null : remoteRegionsToFetch.get().split(","));
    
		//服务注册的过期时间的阈值
        if (config.shouldFetchRegistry()) {
            this.registryStalenessMonitor = new ThresholdLevelsMetric(this, METRIC_REGISTRY_PREFIX + "lastUpdateSec_", new long[]{15L, 30L, 60L, 120L, 240L, 480L});
        } else {
            this.registryStalenessMonitor = ThresholdLevelsMetric.NO_OP_METRIC;
        }
		//配置健康检查的监控程序，15L, 30L, 60L, 120L, 240L, 480L 则代表不同的阈值
        if (config.shouldRegisterWithEureka()) {
            this.heartbeatStalenessMonitor = new ThresholdLevelsMetric(this, METRIC_REGISTRATION_PREFIX + "lastHeartbeatSec_", new long[]{15L, 30L, 60L, 120L, 240L, 480L});
        } else {
            this.heartbeatStalenessMonitor = ThresholdLevelsMetric.NO_OP_METRIC;
        }

        logger.info("Initializing Eureka in region {}", clientConfig.getRegion());
		//如果不需要注册客户端以及不需要拉取注册表，那么就清空回调以及心跳线程
        if (!config.shouldRegisterWithEureka() && !config.shouldFetchRegistry()) {
            logger.info("Client configured to neither register nor query for data.");
            scheduler = null;
            heartbeatExecutor = null;
            cacheRefreshExecutor = null;
            eurekaTransport = null;
            instanceRegionChecker = new InstanceRegionChecker(new PropertyBasedAzToRegionMapper(config), clientConfig.getRegion());

            // This is a bit of hack to allow for existing code using DiscoveryManager.getInstance()
            // to work with DI'd DiscoveryClient
            DiscoveryManager.getInstance().setDiscoveryClient(this);
            DiscoveryManager.getInstance().setEurekaClientConfig(config);

            initTimestampMs = System.currentTimeMillis();
            initRegistrySize = this.getApplications().size();
            registrySize = initRegistrySize;
            logger.info("Discovery Client initialized at timestamp {} with initial instances count: {}",
                    initTimestampMs, initRegistrySize);

            return;  // no need to setup up an network tasks and we are done
        }

        try {
            // 创建定时任务线程池
            scheduler = Executors.newScheduledThreadPool(2,
                    new ThreadFactoryBuilder()
                            .setNameFormat("DiscoveryClient-%d")
                            .setDaemon(true)
                            .build());
			// 创建心跳线程池
            heartbeatExecutor = new ThreadPoolExecutor(
                    1, clientConfig.getHeartbeatExecutorThreadPoolSize(), 0, TimeUnit.SECONDS,
                    new SynchronousQueue<Runnable>(),
                    new ThreadFactoryBuilder()
                            .setNameFormat("DiscoveryClient-HeartbeatExecutor-%d")
                            .setDaemon(true)
                            .build()
            );  // use direct handoff
			//定时拉取注册表的线程池，采用的同步队列
            cacheRefreshExecutor = new ThreadPoolExecutor(
                    1, clientConfig.getCacheRefreshExecutorThreadPoolSize(), 0, TimeUnit.SECONDS,
                    new SynchronousQueue<Runnable>(),
                    new ThreadFactoryBuilder()
                            .setNameFormat("DiscoveryClient-CacheRefreshExecutor-%d")
                            .setDaemon(true)
                            .build()
            );  // use direct handoff
			//创建eureka 请求发起器，主要用于拉取注册表以及发送心跳，注册服务等
            eurekaTransport = new EurekaTransport();
            //初始化 eurekaTransport
            scheduleServerEndpointTask(eurekaTransport, args);

            AzToRegionMapper azToRegionMapper;
            //是否配置dns解析
            if (clientConfig.shouldUseDnsForFetchingServiceUrls()) {
                azToRegionMapper = new DNSBasedAzToRegionMapper(clientConfig);
            } else {
                azToRegionMapper = new PropertyBasedAzToRegionMapper(clientConfig);
            }
            if (null != remoteRegionsToFetch.get()) {
                azToRegionMapper.setRegionsToFetch(remoteRegionsToFetch.get().split(","));
            }
            instanceRegionChecker = new InstanceRegionChecker(azToRegionMapper, clientConfig.getRegion());
        } catch (Throwable e) {
            throw new RuntimeException("Failed to initialize DiscoveryClient!", e);
        }

        if (clientConfig.shouldFetchRegistry()) {
            try {
                boolean primaryFetchRegistryResult = fetchRegistry(false);
                if (!primaryFetchRegistryResult) {
                    logger.info("Initial registry fetch from primary servers failed");
                }
                boolean backupFetchRegistryResult = true;
                if (!primaryFetchRegistryResult && !fetchRegistryFromBackup()) {
                    backupFetchRegistryResult = false;
                    logger.info("Initial registry fetch from backup servers failed");
                }
                if (!primaryFetchRegistryResult && !backupFetchRegistryResult && clientConfig.shouldEnforceFetchRegistryAtInit()) {
                    throw new IllegalStateException("Fetch registry error at startup. Initial fetch failed.");
                }
            } catch (Throwable th) {
                logger.error("Fetch registry error at startup: {}", th.getMessage());
                throw new IllegalStateException(th);
            }
        }

        // call and execute the pre registration handler before all background tasks (inc registration) is started
        if (this.preRegistrationHandler != null) {
            this.preRegistrationHandler.beforeRegistration();
        }

        if (clientConfig.shouldRegisterWithEureka() && clientConfig.shouldEnforceRegistrationAtInit()) {
            try {
                if (!register() ) {
                    throw new IllegalStateException("Registration error at startup. Invalid server response.");
                }
            } catch (Throwable th) {
                logger.error("Registration error at startup: {}", th.getMessage());
                throw new IllegalStateException(th);
            }
        }

        // finally, init the schedule tasks (e.g. cluster resolvers, heartbeat, instanceInfo replicator, fetch
        initScheduledTasks();

        try {
            Monitors.registerObject(this);
        } catch (Throwable e) {
            logger.warn("Cannot register timers", e);
        }

        // This is a bit of hack to allow for existing code using DiscoveryManager.getInstance()
        // to work with DI'd DiscoveryClient
        DiscoveryManager.getInstance().setDiscoveryClient(this);
        DiscoveryManager.getInstance().setEurekaClientConfig(config);

        initTimestampMs = System.currentTimeMillis();
        initRegistrySize = this.getApplications().size();
        registrySize = initRegistrySize;
        logger.info("Discovery Client initialized at timestamp {} with initial instances count: {}",
                initTimestampMs, initRegistrySize);
    }
```



### 1.3 ApplicationInfoManager

根据配置信息创建 **InstanceInfo** 当前客户端的实体信息，其中对服务状态信息的管理是在当前对象中进行

```java
@Bean
@ConditionalOnMissingBean(value = ApplicationInfoManager.class,
                          search = SearchStrategy.CURRENT)
public ApplicationInfoManager eurekaApplicationInfoManager(
    EurekaInstanceConfig config) {
    InstanceInfo instanceInfo = new InstanceInfoFactory().create(config);
    return new ApplicationInfoManager(config, instanceInfo);
}
```

### 1.4 EurekaRegistration

当前 **EurekaRegistration** 并不会在spring启动时就进行自动注册，会在 **EurekaAutoServiceRegistration** 对其进行初始化，除非指定了 **spring.cloud.service-registry.auto-registration** 属性

```java
@Bean
		@ConditionalOnBean(AutoServiceRegistrationProperties.class) //当容器中有当前配置对象时需要进行加载
		@ConditionalOnProperty(
				value = "spring.cloud.service-registry.auto-registration.enabled",
				matchIfMissing = true)
public EurekaRegistration eurekaRegistration(EurekaClient eurekaClient,
                                             CloudEurekaInstanceConfig instanceConfig,
                                             ApplicationInfoManager applicationInfoManager, @Autowired(required = false) ObjectProvider<HealthCheckHandler> healthCheckHandler) {
    //创建Eureka注册对象
    return EurekaRegistration.builder(instanceConfig).with(applicationInfoManager)
        .with(eurekaClient).with(healthCheckHandler).build();
}
```

### 1.5 EurekaAutoServiceRegistration

自动服务注册服务，通过 **ApplicationManager** 管理器对客户端的状态信息进行管理，以及设置了健康检查url地址还有发送状态变更的事件，正在做服务注册是在创建 **EurekaClient** 的时候就已经注册了

![1657617923171](https://cdn.jsdelivr.net/gh/haijun823/note-picture@main/note-picture1657617923171.png)

```java
@Bean
@ConditionalOnBean(AutoServiceRegistrationProperties.class)
@ConditionalOnProperty(
    value = "spring.cloud.service-registry.auto-registration.enabled",
    matchIfMissing = true)
public EurekaAutoServiceRegistration eurekaAutoServiceRegistration(
    ApplicationContext context, EurekaServiceRegistry registry,
    EurekaRegistration registration) {
    //自动服务注册，其中只是修改了服务的状态，因为在 EurekaClient中进行注册时的状态为 START
    return new EurekaAutoServiceRegistration(context, registry, registration);
}
```