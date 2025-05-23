---
title: "@Scheule注解调用过程源码解析"
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
thumbnail: https://cdn2.zzzmh.cn/wallpaper/origin/eea5d2e63820469db675e24dee7b7c36.jpg/fhd?auth_key=1749052800-a9d6db4059f59d93c3c9dff4b20c7e6a2df84469-0-bbad77317514830f5d4b0a9440896c80
published: true
---

# 1. @EnableScheduling

```java
@Target(ElementType.TYPE)
@Retention(RetentionPolicy.RUNTIME)
@Import(SchedulingConfiguration.class) //导入配置文件对象
@Documented
public @interface EnableScheduling {

}
```

# 2. @Scheduled

- fixedDelay：在上一次任务执行结束之后，才会执行
- fixedRate：在上一次任务开始执行的时候开始计算时间，超过时间后，会立即执行

```java
@Target({ElementType.METHOD, ElementType.ANNOTATION_TYPE})
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Repeatable(Schedules.class)
public @interface Scheduled {

	/**
	 * 禁用特殊corn表达式的值
	 */
	String CRON_DISABLED = ScheduledTaskRegistrar.CRON_DISABLED;


	/**
	 * corn表达式值
	 */
	String cron() default "";

	/**
	 * corn表达的时区，如果为空，将使用服务器的本地时区
	 */
	String zone() default "";

	/**
	 * 每隔 fixedDelay 的时间就会执行一次，例如：1000，1秒钟执行一次，在上一次任务结束后1秒开始执行
	 */
	long fixedDelay() default -1;

	/**
	 * 返回字符串的形式，跟fixedDelay只能存在一个
	 */
	String fixedDelayString() default "";

	/**
	 * 每隔 fixedRate 的时间就会执行一次，例如：1000，1秒钟执行一次，如果上一次任务执行超过一秒，那么会立即执行下一次任务
	 */
	long fixedRate() default -1;

	/**
	 * 返回字符串的形式
	 */
	String fixedRateString() default "";

	/**
	 * 初始化延时多久开始执行
	 */
	long initialDelay() default -1;

	/**
	 * 返回字符串的形式，跟 initialDelay 只能存在一个
	 */
	String initialDelayString() default "";

	/**
	 * 默认毫秒
	 */
	TimeUnit timeUnit() default TimeUnit.MILLISECONDS;

}
```



# 3. SchedulingConfiguration

```java
@Configuration(proxyBeanMethods = false)
@Role(BeanDefinition.ROLE_INFRASTRUCTURE)
public class SchedulingConfiguration {
	/**
	 * 配置一个Bean的后置处理器，ScheduledAnnotationBeanPostProcessor 实现了 SmartInitializingSingleton、 ApplicationListener<ContextRefreshedEvent>、MergedBeanDefinitionPostProcessor
	 * SmartInitializingSingleton：在bean创建完之后，会执行调用；
	 * ApplicationListener<ContextRefreshedEvent>：在容器创建完之后进行刷新时发布事件
	 * MergedBeanDefinitionPostProcessor：在bean执行初始化完成之后，执行当前接口的 postProcessAfterInitialization() 用于扫描所有有 @Scheduled、@Schedules注解的bean
	 *
	 * @return
	 */
	@Bean(name = TaskManagementConfigUtils.SCHEDULED_ANNOTATION_PROCESSOR_BEAN_NAME)
	@Role(BeanDefinition.ROLE_INFRASTRUCTURE)
	public ScheduledAnnotationBeanPostProcessor scheduledAnnotationProcessor() {
		return new ScheduledAnnotationBeanPostProcessor();
	}

}
```



# 4. ScheduledAnnotationBeanPostProcessor

## 4.1 postProcessAfterInitialization

- 判断是否是Aop相关指定的类型
- 判断当前bean是否是包含在无注解的集合中
- 将 Scheduled、Schedules注解进行合并，返回一个Map<Method, Set<Scheduled>> 进行数据的处理
- 最后循环遍历调用 processScheduled() 方法

```java
@Override
	public Object postProcessAfterInitialization(Object bean, String beanName) {
		//忽略掉下面的类
		if (bean instanceof AopInfrastructureBean || bean instanceof TaskScheduler ||
				bean instanceof ScheduledExecutorService) {
			return bean;
		}

		//获取到bean的class类
		Class<?> targetClass = AopProxyUtils.ultimateTargetClass(bean);
		//判断当前bean是否没有注解
		if (!this.nonAnnotatedClasses.contains(targetClass) &&
			//判断是否拥有下面的注解
				AnnotationUtils.isCandidateClass(targetClass, Arrays.asList(Scheduled.class, Schedules.class))) {
			//将 Scheduled、Schedules两个注解进行合并成一个 Set<Scheduled> 集合
			Map<Method, Set<Scheduled>> annotatedMethods = MethodIntrospector.selectMethods(targetClass,
					(MethodIntrospector.MetadataLookup<Set<Scheduled>>) method -> {
						Set<Scheduled> scheduledAnnotations = AnnotatedElementUtils.getMergedRepeatableAnnotations(
								method, Scheduled.class, Schedules.class);
						return (!scheduledAnnotations.isEmpty() ? scheduledAnnotations : null);
					});
			//如果没有注解则存入 nonAnnotatedClasses中标记当前类没有注解可以处理
			if (annotatedMethods.isEmpty()) {
				this.nonAnnotatedClasses.add(targetClass);
				if (logger.isTraceEnabled()) {
					logger.trace("No @Scheduled annotations found on bean class: " + targetClass);
				}
			}
			else {
				//如果有注解循环遍历进行处理
				annotatedMethods.forEach((method, scheduledAnnotations) ->
						scheduledAnnotations.forEach(scheduled -> processScheduled(scheduled, method, bean)));
				if (logger.isTraceEnabled()) {
					logger.trace(annotatedMethods.size() + " @Scheduled methods processed on bean '" + beanName +
							"': " + annotatedMethods);
				}
			}
		}
		return bean;
	}
```

## 4.2 processScheduled

