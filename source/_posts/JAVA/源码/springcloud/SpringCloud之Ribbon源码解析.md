---
title: SpringCloud之Ribbon源码解析
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
thumbnail: https://images.unsplash.com/photo-1582719202047-76d3432ee323?crop=entropy&cs=srgb&fm=jpg&ixid=M3w2NDU1OTF8MHwxfHJhbmRvbXx8fHx8fHx8fDE3NDMxNTE4MjB8&ixlib=rb-4.0.3&q=85&w=1920&h=1080
published: false
---

# Ribbon

基于 **Ribbon 2.2.6.RELEASE版本进行源码阅读**

```xml
<dependency>
    <groupId>org.springframework.cloud</groupId>
    <artifactId>spring-cloud-netflix-ribbon</artifactId>
    <version>2.2.6.RELEASE</version>
</dependency>
```

一键拉取 **spring-cloud-common、spring-cloud-alibaba、spring-cloud-netflix、spring-cloud-openfeign** 源码

> https://gitee.com/haijun1998/spring-cloud-git.git

## 1. @RibbonClients

```java
@Import(RibbonClientConfigurationRegistrar.class)
public @interface RibbonClients {
	
	/**
	 * 配置ribbon客户端的信息
	 * 
	 * @return
	 */
	RibbonClient[] value() default {};

	/**
	 * 默认的配置类
	 * 
	 * @return
	 */
	Class<?>[] defaultConfiguration() default {};

}
```

### 1.1 RibbonClientConfigurationRegistrar

**RibbonClientConfigurationRegistrar** 所要做的就是将配置客户端配置文件注册到容器当中去，在后续使用时直接通过对应的配置信息拉取

```java
public class RibbonClientConfigurationRegistrar implements ImportBeanDefinitionRegistrar {

	@Override
	public void registerBeanDefinitions(AnnotationMetadata metadata,
			BeanDefinitionRegistry registry) {
		Map<String, Object> attrs = metadata
				.getAnnotationAttributes(RibbonClients.class.getName(), true);
        //获取到配置的 RibbonClient 注解
		if (attrs != null && attrs.containsKey("value")) {
			AnnotationAttributes[] clients = (AnnotationAttributes[]) attrs.get("value");
			for (AnnotationAttributes client : clients) {
				registerClientConfiguration(registry, getClientName(client),
						client.get("configuration"));
			}
		}
        //获取到是否存在默认的配置类，将默认配置文件注入到容器中
		if (attrs != null && attrs.containsKey("defaultConfiguration")) {
			String name;
			if (metadata.hasEnclosingClass()) {
				name = "default." + metadata.getEnclosingClassName();
			}
			else {
				name = "default." + metadata.getClassName();
			}
			registerClientConfiguration(registry, name,
					attrs.get("defaultConfiguration"));
		}
        //处理 RibbonClient 注解
		Map<String, Object> client = metadata
				.getAnnotationAttributes(RibbonClient.class.getName(), true);
		String name = getClientName(client);
		if (name != null) {
			registerClientConfiguration(registry, name, client.get("configuration"));
		}
	}
}
```

## 2. 配置类

- RibbonAutoConfiguration
- RibbonClientConfiguration

## 3. SpringClientFactory

跟 **OpenFeign** 中的 **FeignContext** 类一样的实现方式通过实现了 **Spring Cloud Common** 中提供的 **NamedContextFactory<RibbonClientSpecification\>** 一个客户端对应一个子容器的方式进行环境隔离，这个类是跟 **Feign** 进行交互的核心类，**Feign** 中通过 **CachingSpringLoadBalancerFactory** 对其进行包装

```java
public class SpringClientFactory extends NamedContextFactory<RibbonClientSpecification> {
 
    /**
	 * 从容器中获取到实现的客户端，feign 中的实现是 FeignLoadBalancer，根据实现方式不同客户端也不同
	 * httpclient：RibbonLoadBalancingHttpClient 进行装配
	 * okhttp：OkHttpLoadBalancingClient
	 * 默认客户端：RestClient
	 */
	public <C extends IClient<?, ?>> C getClient(String name, Class<C> clientClass) {
		return getInstance(name, clientClass);
	}

	/**
	 * 获取对应客户端容器中查询出 ILoadBalancer 的实现类用于负载获取，子容器获取不到就从父容器中获取
	 */
	public ILoadBalancer getLoadBalancer(String name) {
		return getInstance(name, ILoadBalancer.class);
	}

	/**
	 * 从对应的名称容器中获取到配置对象
	 */
	public IClientConfig getClientConfig(String name) {
		return getInstance(name, IClientConfig.class);
	}
}
```

