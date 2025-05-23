---
title: "@Qualified注解原理解析"
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
thumbnail: https://cdn2.zzzmh.cn/wallpaper/origin/4e952013c980414da49ee18e7fb4199b.jpg/fhd?auth_key=1749052800-a9d6db4059f59d93c3c9dff4b20c7e6a2df84469-0-6a2ffb5ffc3b4a6f9912f1b6c7d68a36
published: true
---
# @Qualified

在spring中进行依赖注入的方式有两个注解可以使用，分别是 **@Resource、@Autowired** 两个注解其对应的功能分别是

- Resource：默认按照名称进行装配，可以通过name属性指定名称，如果没有指定name属性，当注解写在字段上时，默认取字段名进行查找注入，如果写在setter方法上默认取属性名进行装配。当找不到与名称匹配的bean时才按照类型进行装配。但是需要注意的是，如果name属性一旦指定，就只会按照名称进行装配 
- Autowired： 默认按类型装配，默认情况下必须要求依赖对象必须存在。如果 Spring 没有其他提示，将会按照需要注入的变量名称来寻找合适的 bean 

现在有个场景就是如果我一个接口下面有多个实现，我主要想注入标记了对应注解的实现类，上面的两个注解就不能单独进行使用了，需要进行搭配使用；例如：**@LoadBalanced** 注解实现负载均衡；

```java
@Target({ ElementType.FIELD, ElementType.PARAMETER, ElementType.METHOD })
@Retention(RetentionPolicy.RUNTIME)
@Documented
@Inherited
@Qualifier
public @interface LoadBalanced {

}
```

注意 **@LoadBalanced** 注解中包含了一个 **@Qualifier** 的注解，这个注解的用处就是可以在一个接口多个实现类中进行指定注入例如：spring cloud提供的对 **RestTemplate** 进行负载，下面我们看一下具体的源码是如何进行导入的

```java
public class LoadBalancerAutoConfiguration {

	@LoadBalanced
	@Autowired(required = false)
	private List<RestTemplate> restTemplates = Collections.emptyList();
}
```

spring工厂在实例化后对属性进行赋值时执行 **InstantiationAwareBeanPostProcessor** 接口的实现类

而进行属性注入的时候是通过 **AutowiredAnnotationBeanPostProcessor** 处理器在bean实例化之后对 **@Autowired** 注解进行属性赋值的

```java
public PropertyValues postProcessProperties(PropertyValues pvs, Object bean, String beanName) {
		/**
		 * 获取到对象内部中 是否有Autowired注解标注的属性
		 * InjectionMetadata：
		 * 包含一个list的 InjectedElement接口元素，会将 @Autowired注解标记的字段或者方法解析成对应的实现类
		 */
		InjectionMetadata metadata = findAutowiringMetadata(beanName, bean.getClass(), pvs);
		try {
			metadata.inject(bean, beanName, pvs);
		}
		catch (BeanCreationException ex) {
			throw ex;
		}
		catch (Throwable ex) {
			throw new BeanCreationException(beanName, "Injection of autowired dependencies failed", ex);
		}
		return pvs;
	}
```

下面的代码就是循环调用 **InjectedElement** 的方法

```java
public void inject(Object target, @Nullable String beanName, @Nullable PropertyValues pvs) throws Throwable {
		Collection<InjectedElement> checkedElements = this.checkedElements;
		Collection<InjectedElement> elementsToIterate =
				(checkedElements != null ? checkedElements : this.injectedElements);
		if (!elementsToIterate.isEmpty()) {
			for (InjectedElement element : elementsToIterate) {
				element.inject(target, beanName, pvs);
			}
		}
	}
```

下面以 **AutowiredFieldElement** 为例子进行源码的解读。下面的代码可以看到最终解析是通过 **resolveFieldValue()** 的方法进行解析

```java
protected void inject(Object bean, @Nullable String beanName, @Nullable PropertyValues pvs) throws Throwable {
			//强转为字段对象
			Field field = (Field) this.member;
			Object value;
			//是否被缓存了，默认false
			if (this.cached) {
				try {
					//开启了缓存直接解析已经缓存了的value值
					value = resolvedCachedArgument(beanName, this.cachedFieldValue);
				}
				catch (NoSuchBeanDefinitionException ex) {
					// Unexpected removal of target bean for cached argument -> re-resolve
					value = resolveFieldValue(field, bean, beanName);
				}
			}
			else {
				//通过字段、bean、bean名称去bean工厂中解析出对应的依赖值
				value = resolveFieldValue(field, bean, beanName);
			}
			if (value != null) {
				ReflectionUtils.makeAccessible(field);
				field.set(bean, value);
			}
		}
```

解析字段的依赖

```java
@Nullable
		private Object resolveFieldValue(Field field, Object bean, @Nullable String beanName) {
			//依赖描述类，里面包括了 Method、Filed 的描述
			DependencyDescriptor desc = new DependencyDescriptor(field, this.required);
			//设置字段或者方法所属的bean对象类
			desc.setContainingClass(bean.getClass());
			//获取到的依赖bean的名称
			Set<String> autowiredBeanNames = new LinkedHashSet<>(1);
			Assert.state(beanFactory != null, "No BeanFactory available");
			//获取到spring工厂定义的类型转换器
			TypeConverter typeConverter = beanFactory.getTypeConverter();
			Object value;
			try {
				//bean工厂解析出依赖
				value = beanFactory.resolveDependency(desc, beanName, autowiredBeanNames, typeConverter);
			}
        }
```

当前方法对需要注入的依赖类型进行判断，最重要的是 **doResolveDependency()** 方法

```java
public Object resolveDependency(DependencyDescriptor descriptor, @Nullable String requestingBeanName,
			@Nullable Set<String> autowiredBeanNames, @Nullable TypeConverter typeConverter) throws BeansException {

		descriptor.initParameterNameDiscovery(getParameterNameDiscoverer());
		//处理 Optional
		if (Optional.class == descriptor.getDependencyType()) {
			return createOptionalDependency(descriptor, requestingBeanName);
		}
		//处理 ObjectFactory 类
		else if (ObjectFactory.class == descriptor.getDependencyType() ||
				ObjectProvider.class == descriptor.getDependencyType()) {
			return new DependencyObjectProvider(descriptor, requestingBeanName);
		}
		else if (javaxInjectProviderClass == descriptor.getDependencyType()) {
			return new Jsr330Factory().createDependencyProvider(descriptor, requestingBeanName);
		}
		else {
			//获取到是否是懒加载，通过 ContextAnnotationAutowireCandidateResolver 进行判断是否有 @Lazy注解，如果是懒加载会通过 ProxyFactory 创建一个动态代理类
			Object result = getAutowireCandidateResolver().getLazyResolutionProxyIfNecessary(
					descriptor, requestingBeanName);
			if (result == null) {
				result = doResolveDependency(descriptor, requestingBeanName, autowiredBeanNames, typeConverter);
			}
			return result;
		}
	}
```

**doResolveDependency()** 方法中最主要调用的是下面两个方法

- resolveMultipleBeans：解析list、array、map等依赖
- findAutowireCandidates：解析单个bean的依赖方法

整个方法所做的事情就是，先检查依赖缓存中是否已经缓存了字段的依赖数据，如果有的话直接解析缓存数据，没有的话就先解析所需要的字段依赖是集合相关的还是单个bean，优先解析字段依赖的类型是 **list、array、map**，然后再解析单个

```java
public Object doResolveDependency(DependencyDescriptor descriptor, @Nullable String beanName,
			@Nullable Set<String> autowiredBeanNames, @Nullable TypeConverter typeConverter) throws BeansException {

		//设置了一下当前正在解析的注入参数，通过本地线程进行保存
		InjectionPoint previousInjectionPoint = ConstructorResolver.setCurrentInjectionPoint(descriptor);
		try {
			Object shortcut = descriptor.resolveShortcut(this);
			if (shortcut != null) {
				return shortcut;
			}

			Class<?> type = descriptor.getDependencyType();
			/**
			 * 通过当前指定的自动注入的接口器来解析需要解析的bean的名称，目前有两个实现类
			 * QualifierAnnotationAutowireCandidateResolver：用于处理 @Value 、@Qualifier 注解
			 * SimpleAutowireCandidateResolver，默认的解析器，并没有实现什么特殊的功能
			 * ContextAnnotationAutowireCandidateResolver：继承至 QualifierAnnotationAutowireCandidateResolver
			 * 这里回去找 @Value 注解是否存在
			 */
			Object value = getAutowireCandidateResolver().getSuggestedValue(descriptor);
			if (value != null) {
				//如果value不为空的话进行下面的的处理
				.....代码省略
			}
			//解析依赖的是否是map、list、数组等
			Object multipleBeans = resolveMultipleBeans(descriptor, beanName, autowiredBeanNames, typeConverter);
			if (multipleBeans != null) {
				return multipleBeans;
			}
			//获取到匹配的bean
			Map<String, Object> matchingBeans = findAutowireCandidates(beanName, type, descriptor);
			//如果依赖都没有找找到，先判断是否是 required=true
			if (matchingBeans.isEmpty()) {
				if (isRequired(descriptor)) {
					//抛出bean依赖没有找到的问题
					raiseNoMatchingBeanFound(type, descriptor.getResolvableType(), descriptor);
				}
				return null;
			}
			String autowiredBeanName;
			Object instanceCandidate;
			//匹配的bean的个数如果大于1
			if (matchingBeans.size() > 1) {
				//查看是否标注了 @Primary注解、@priority，优先使用注入
				autowiredBeanName = determineAutowireCandidate(matchingBeans, descriptor);
				if (autowiredBeanName == null) {
					if (isRequired(descriptor) || !indicatesMultipleBeans(type)) {
						return descriptor.resolveNotUnique(descriptor.getResolvableType(), matchingBeans);
					}
					else {
						return null;
					}
				}
				instanceCandidate = matchingBeans.get(autowiredBeanName);
			}
			else {
				Map.Entry<String, Object> entry = matchingBeans.entrySet().iterator().next();
				//如果只有一个直接获取key值
				autowiredBeanName = entry.getKey();
				instanceCandidate = entry.getValue();
			}

			if (autowiredBeanNames != null) {
				autowiredBeanNames.add(autowiredBeanName);
			}
			if (instanceCandidate instanceof Class) {
				//如果类型为class，直接去bean工厂中调用 getBean() 查询，如果没有查询到直接实例化
				instanceCandidate = descriptor.resolveCandidate(autowiredBeanName, type, this);
			}
			Object result = instanceCandidate;
			if (result instanceof NullBean) {
				if (isRequired(descriptor)) {
					raiseNoMatchingBeanFound(type, descriptor.getResolvableType(), descriptor);
				}
				result = null;
			}
			if (!ClassUtils.isAssignableValue(type, result)) {
				throw new BeanNotOfRequiredTypeException(autowiredBeanName, type, instanceCandidate.getClass());
			}
			return result;
		}
		finally {
			ConstructorResolver.setCurrentInjectionPoint(previousInjectionPoint);
		}
	}
```

**resolveMultipleBeans()** 方法解析多元素，其中会进行判断字段的数据类型是什么，**array、map、collection、interface** 等组装的方式不一样但是获取数据bean的方式都一样，都是通过 **findAutowireCandidates()** 方法去通过bean工厂获取数据

```java
private Object resolveMultipleBeans(DependencyDescriptor descriptor, @Nullable String beanName,
			@Nullable Set<String> autowiredBeanNames, @Nullable TypeConverter typeConverter) {
		//获取到当前字段需要依赖的类型
		Class<?> type = descriptor.getDependencyType();
		//处理 StreamDependencyDescriptor 类型的依赖描述器，这里一般是 DependencyDescriptor
		if (descriptor instanceof StreamDependencyDescriptor) {
			return stream;
		}
		//判断依赖的类型是否是数组类型
		else if (type.isArray()) {
			Class<?> componentType = type.getComponentType();
			ResolvableType resolvableType = descriptor.getResolvableType();
			//解析数组依赖的类型是什么
			Class<?> resolvedArrayType = resolvableType.resolve(type);
			if (resolvedArrayType != type) {
				componentType = resolvableType.getComponentType().resolve();
			}
			if (componentType == null) {
				return null;
			}
			//获取到适合的依赖，通过 MultiElementDescriptor 对依赖描述器封装成多元素描述器
			Map<String, Object> matchingBeans = findAutowireCandidates(beanName, componentType,
					new MultiElementDescriptor(descriptor));
			if (matchingBeans.isEmpty()) {
				return null;
			}
			if (autowiredBeanNames != null) {
				autowiredBeanNames.addAll(matchingBeans.keySet());
			}
			//通过 TypeConverter 来进行转换判断类型之间是否可以互相转换
			TypeConverter converter = (typeConverter != null ? typeConverter : getTypeConverter());
			Object result = converter.convertIfNecessary(matchingBeans.values(), resolvedArrayType);
			if (result instanceof Object[]) {
				Comparator<Object> comparator = adaptDependencyComparator(matchingBeans);
				if (comparator != null) {
					Arrays.sort((Object[]) result, comparator);
				}
			}
			return result;
		}
		//集合类型并且是接口
		else if (Collection.class.isAssignableFrom(type) && type.isInterface()) {
			....省略
		}
		else if (Map.class == type) {
			//如果要求的参数为map，首先获取到 Map<key,value> 泛型的类型，通过泛型key和value的类型到容器中查找对应的依赖
			....省略
		}
		else {
			return null;
		}
	}
```

**findAutowireCandidates()** 第一步就是直接通过依赖的类型去bean工厂中直接获取到对应名称的依赖，然后通过后面的 **isAutowireCandidate()** 方法进行判断是否存在 **@Qualified** 注解，所用到的

```java
protected Map<String, Object> findAutowireCandidates(
			@Nullable String beanName, Class<?> requiredType, DependencyDescriptor descriptor) {
		//根据类型会回去到所有合适的bean的名称
		String[] candidateNames = BeanFactoryUtils.beanNamesForTypeIncludingAncestors(
				this, requiredType, true, descriptor.isEager());
		//创建一个LinkedHashMap
		Map<String, Object> result = CollectionUtils.newLinkedHashMap(candidateNames.length);
		//首先从已经解析了的依赖缓存池中查找对应的值是否有
		for (Map.Entry<Class<?>, Object> classObjectEntry : this.resolvableDependencies.entrySet()) {
			//还是先解析缓存，代码省略
		}
		//处理找到的适合的bean
		for (String candidate : candidateNames) {
			//判断是否是自己引用自己，以及通过 isAutowireCandidate() 调用 QualifierAnnotationAutowireCandidateResolver.isAutowireCandidate() 判断是否有 @Qualifier
			if (!isSelfReference(beanName, candidate) && isAutowireCandidate(candidate, descriptor)) {
				//将查找出的bean名称通过工厂对象直接进行解析然后添加到 Map中
				addCandidateEntry(result, candidate, descriptor, requiredType);
			}
		}
		//只有上面步骤 以及解析的依赖和判断不含@Qualifier注解时，解析的依赖还是为空时走这个分支
		if (result.isEmpty()) {
			//判断要求的类是否是一个接口或者list
			boolean multiple = indicatesMultipleBeans(requiredType);
			// Consider fallback matches if the first pass failed to find anything...
			DependencyDescriptor fallbackDescriptor = descriptor.forFallbackMatch();
			for (String candidate : candidateNames) {
				//通过 AutowireCandidateResolver 类解析当前依赖是否有 @Qualifier注解通过名称进行注入 QualifierAnnotationAutowireCandidateResolver
				if (!isSelfReference(beanName, candidate) && isAutowireCandidate(candidate, fallbackDescriptor) &&
						(!multiple || getAutowireCandidateResolver().hasQualifier(descriptor))) {
					addCandidateEntry(result, candidate, descriptor, requiredType);
				}
			}
			if (result.isEmpty() && !multiple) {
				for (String candidate : candidateNames) {
					if (isSelfReference(beanName, candidate) &&
							(!(descriptor instanceof MultiElementDescriptor) || !beanName.equals(candidate)) &&
							isAutowireCandidate(candidate, fallbackDescriptor)) {
						addCandidateEntry(result, candidate, descriptor, requiredType);
					}
				}
			}
		}
		return result;
	}
```

 **isAutowireCandidate()** 搜先判断bean定义是否存在，再判断bean是否存在于单例池中，最后调用是通过 **AutowireCandidateResolver** 的子实现类 **ContextAnnotationAutowireCandidateResolver** 进行判断是否存在 **@Qualified** 注解。

 **ContextAnnotationAutowireCandidateResolver** 的配置是通过 **AnnotatedBeanDefinitionReader** 类在spring工厂初始化时就进行了设置

```java
protected boolean isAutowireCandidate(
			String beanName, DependencyDescriptor descriptor, AutowireCandidateResolver resolver)
			throws NoSuchBeanDefinitionException {

		String bdName = BeanFactoryUtils.transformedBeanName(beanName);
		//bean定义中是否存在bean名称
		if (containsBeanDefinition(bdName)) {
			return isAutowireCandidate(beanName, getMergedLocalBeanDefinition(bdName), descriptor, resolver);
		}
		//是否存在于单例池中
		else if (containsSingleton(beanName)) {
			return isAutowireCandidate(beanName, new RootBeanDefinition(getType(beanName)), descriptor, resolver);
		}

		BeanFactory parent = getParentBeanFactory();
		if (parent instanceof DefaultListableBeanFactory) {
			return ((DefaultListableBeanFactory) parent).isAutowireCandidate(beanName, descriptor, resolver);
		}
		else if (parent instanceof ConfigurableListableBeanFactory) {
			return ((ConfigurableListableBeanFactory) parent).isAutowireCandidate(beanName, descriptor);
		}
		else {
			return true;
		}
	}
```

```java
protected boolean isAutowireCandidate(String beanName, RootBeanDefinition mbd,
			DependencyDescriptor descriptor, AutowireCandidateResolver resolver) {
		//转换bean名称
		String bdName = BeanFactoryUtils.transformedBeanName(beanName);
		//解析bean的class
		resolveBeanClass(mbd, bdName);
		if (mbd.isFactoryMethodUnique && mbd.factoryMethodToIntrospect == null) {
			new ConstructorResolver(this).resolveFactoryMethodIfPossible(mbd);
		}
		//创建一个  BeanDefinitionHolder 对象后面用于处理
		BeanDefinitionHolder holder = (beanName.equals(bdName) ?
				this.mergedBeanDefinitionHolders.computeIfAbsent(beanName,
						key -> new BeanDefinitionHolder(mbd, beanName, getAliases(bdName))) :
				new BeanDefinitionHolder(mbd, beanName, getAliases(bdName)));
		//通过子类 ContextAnnotationAutowireCandidateResolver 进行判断
		return resolver.isAutowireCandidate(holder, descriptor);
	}
```

 **ContextAnnotationAutowireCandidateResolver** 继承至 **QualifierAnnotationAutowireCandidateResolver**；最重要的逻辑判断就是在 **checkQualifiers()** 方法中

```java
public boolean isAutowireCandidate(BeanDefinitionHolder bdHolder, DependencyDescriptor descriptor) {
		boolean match = super.isAutowireCandidate(bdHolder, descriptor);
		if (match) {
			//检查@Qualifiers注解
			match = checkQualifiers(bdHolder, descriptor.getAnnotations());
            //如果匹配了在判断方法的参数
			if (match) {
				MethodParameter methodParam = descriptor.getMethodParameter();
				if (methodParam != null) {
					Method method = methodParam.getMethod();
					if (method == null || void.class == method.getReturnType()) {
						match = checkQualifiers(bdHolder, methodParam.getMethodAnnotations());
					}
				}
			}
		}
		return match;
	}
```

先通过 **isQualifier()** 判断当前字段是否标记了 **@Qualified** 注解，然后再通过 **checkQualifier()** 去检查获取到的bean依赖对象中是否也含有 **@Qualified** 注解

```java
protected boolean checkQualifiers(BeanDefinitionHolder bdHolder, Annotation[] annotationsToSearch) {
		if (ObjectUtils.isEmpty(annotationsToSearch)) {
			return true;
		}
		SimpleTypeConverter typeConverter = new SimpleTypeConverter();
    	//遍历获取到字段的注解
		for (Annotation annotation : annotationsToSearch) {
			Class<? extends Annotation> type = annotation.annotationType();
			boolean checkMeta = true;
			boolean fallbackToMeta = false;
			//判断注解是否有 @Qualifier 注解
			if (isQualifier(type)) {
				//判断获取到的依赖bean对象是否也有 @Qualifier注解
				if (!checkQualifier(bdHolder, annotation, typeConverter)) {
					fallbackToMeta = true;
				}
				else {
					checkMeta = false;
				}
			}
            ......省略
		}
		return true;
	}
```

通过上面的方法我们可以看到，先获取到 **字段中所有的注解，然后遍历注解**；通过 **checkQualifier()** 每一个注解都进行判断是否有  **@Qualified** 例如下面的例子：

自动配置类中需要注入一个 **List<RestTemplate\>** 的类型，下面配置了一个 **RestTemplate** 的bean对象，首先解析字段 **restTemplates** 需要依赖的是一个 **List<RestTemplate\>** 从bean工厂中获取到 **restTemplate1、restTemplate2** 两个bean对象，但是现在字段 **restTemplates** 标记了 **@LoadBalanced** 注解而下面 **checkQualifier()** 的方法就是，先获取到 **restTemplates** 所有的注解 **@LoadBalanced、@Autowired(required = false)** 然后遍历判断是否存在 **@Qualified** ，我们知道 **@LoadBalanced** 中包含了这个注解所有就继续执行，通过当前注解去判断 **restTemplate1、restTemplate2** 两个bean是否也有  **@LoadBalanced** 如果有就符合条件

```java
public class LoadBalancerAutoConfiguration {

	@LoadBalanced
	@Autowired(required = false)
	private List<RestTemplate> restTemplates = Collections.emptyList();
    
    @Bean
    @LoadBalanced
	public RestTemplate restTemplate1() {
		return new RestTemplate();
	}
    
    @Bean
	public RestTemplate restTemplate2() {
		return new RestTemplate();
	}
}
```



```java
protected boolean checkQualifier(
			BeanDefinitionHolder bdHolder, Annotation annotation, TypeConverter typeConverter) {
		//获取到@LoadBalanced 注解
		Class<? extends Annotation> type = annotation.annotationType();
		//获取到bean定义
		RootBeanDefinition bd = (RootBeanDefinition) bdHolder.getBeanDefinition();
		//根据注解的名称获取到适合的限定器
		AutowireCandidateQualifier qualifier = bd.getQualifier(type.getName());
		if (qualifier == null) {
			//通过短名称来获取
			qualifier = bd.getQualifier(ClassUtils.getShortName(type));
		}
		if (qualifier == null) {
			// 先检查当前bean的注解中是否有对应的 @Qualified注解，例如：@LoadBalanced 其中包含@Qualified注解，这里就会进行判断对应的依赖属性是否也有 @LoadBalanced 注解
			Annotation targetAnnotation = getQualifiedElementAnnotation(bd, type);
			// Then, check annotation on factory method, if applicable
			if (targetAnnotation == null) {
				//解析工厂方法上面是否有对应的注解
				targetAnnotation = getFactoryMethodAnnotation(bd, type);
			}
			if (targetAnnotation == null) {
				RootBeanDefinition dbd = getResolvedDecoratedDefinition(bd);
				if (dbd != null) {
					targetAnnotation = getFactoryMethodAnnotation(dbd, type);
				}
			}
			//如果上面获取到的目标注解还是为空的，那么直接获取到的class对象，通过class对象来进行对应注解的获取
			if (targetAnnotation == null) {
				if (getBeanFactory() != null) {
					try {
						Class<?> beanType = getBeanFactory().getType(bdHolder.getBeanName());
						if (beanType != null) {
							targetAnnotation = AnnotationUtils.getAnnotation(ClassUtils.getUserClass(beanType), type);
						}
					}
					catch (NoSuchBeanDefinitionException ex) {
					}
				}
				if (targetAnnotation == null && bd.hasBeanClass()) {
					targetAnnotation = AnnotationUtils.getAnnotation(ClassUtils.getUserClass(bd.getBeanClass()), type);
				}
			}
			if (targetAnnotation != null && targetAnnotation.equals(annotation)) {
				return true;
			}
		}

		......省略
		return true;
	}
```

总结：spring通过在解析字段类型之后从bean工厂中获取到所有适合的依赖bean，然后通过  **QualifierAnnotationAutowireCandidateResolver** 解析器来判断依赖中是否存在对应的 **Qualified** 注解；而我们给 **RestTemplate** 标记的 **@LoadBalanced** 其实本身并没有进行处理，而是通过内部包含的 **@Qualified** 注解进行判断所依赖的bean是否也包含了 **@LoadBalanced**注解；注意点是spring先判断是否存在 **@Qualified** 注解，然后再拿  **@LoadBalanced** 注解去获取到bean的注解