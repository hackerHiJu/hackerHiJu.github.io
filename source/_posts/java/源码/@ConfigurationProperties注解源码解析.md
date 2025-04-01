---
title: "@ConfigurationProperties注解源码解析"
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
thumbnail: https://images.unsplash.com/photo-1682685797857-97de838c192e?crop=entropy&cs=srgb&fm=jpg&ixid=M3w2NDU1OTF8MHwxfHJhbmRvbXx8fHx8fHx8fDE3NDMxNTE3NjN8&ixlib=rb-4.0.3&q=85&w=1920&h=1080
published: false
---

# SpringBoot处理@ConfigurationProperties

探究 SpringBoot 自动注入配置文件对象的源码处理

## 1. EnableConfigurationProperties

注解 **@EnableConfigurationProperties** 用于注入对应的配置文件对象，以及开启配置注入

```java
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@Documented
//导入相关处理类
@Import(EnableConfigurationPropertiesRegistrar.class)
public @interface EnableConfigurationProperties {

	/**
	 * 定义bean的验证器
	 */
	String VALIDATOR_BEAN_NAME = "configurationPropertiesValidator";

	/**
	 * 需要开启配置注入的对象
	 */
	Class<?>[] value() default {};

}
```

## 2. ConfigurationProperties

标记配置类的注解

```java
@Target({ ElementType.TYPE, ElementType.METHOD })
@Retention(RetentionPolicy.RUNTIME)
@Documented
public @interface ConfigurationProperties {

	/**
	 * 别名
	 */
	@AliasFor("prefix")
	String value() default "";

	/**
	 * 前缀
	 */
	@AliasFor("value")
	String prefix() default "";

	/**
	 * 是否忽略字段验证失败
	 */
	boolean ignoreInvalidFields() default false;

	/**
	 * 是否忽略没有找到属性的字段
	 */
	boolean ignoreUnknownFields() default true;

}
```

