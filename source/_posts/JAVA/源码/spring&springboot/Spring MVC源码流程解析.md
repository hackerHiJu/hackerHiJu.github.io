---
title: Spring MVC源码流程解析
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
thumbnail: https://cdn2.zzzmh.cn/wallpaper/origin/d4b96518882111ebb6edd017c2d2eca2.jpg/fhd?auth_key=1749052800-a9d6db4059f59d93c3c9dff4b20c7e6a2df84469-0-6f513c5a33fc378ef8fdb3b94f27dbf8
published: false
---
# Spring MVC源码流程

## 1. Spring Boot挂载DispatcherServlet？

在 **Springboot** 启动时我们都知道内部内置了一个 **tomcat** 的容器，然后将 **DispatcherServlet** 挂载到了容器的内部从而实现了 **Spring MVC** 机制，下面的链接说明了如何将 **DispatcherServlet** 挂载到了 **Spring Boot** 中

[SpringBoot如何挂载DispatcherServlet](https://blog.csdn.net/weixin_43915643/article/details/125421616)



## 2. 配置类

### 2.1 WebMvcConfigurationSupport

spring mvc的配置类 **@EnableWebMvc** 注解会导入的配置类，用于注入spring mvc中的一些核心的配置类，其中有的一些核心配置方法如下：

- requestMappingHandlerMapping：用于创建 **RequestMappingHandlerMapping** 用于处理 **@RequestMapping** 注解的类
- mvcPatternParser：创建 **PathPatternParser** 路径表达式解析器
- mvcUrlPathHelper：创建 **UrlPathHelper** url路径解析器，用于解析http请求的路径
- mvcPathMatcher：创建 **PathMatcher** 路径匹配器
- viewControllerHandlerMapping：用于创建视图控制器的方法，创建的类为 **SimpleUrlHandlerMapping**
- requestMappingHandlerAdapter：创建处理器的适配器，比较重要 **RequestMappingHandlerAdapter**
- httpRequestHandlerAdapter：http请求的处理器适配器，**HttpRequestHandlerAdapter**

### 2.2 WebMvcConfigurer

```java
public interface WebMvcConfigurer {

	/**
	 * 配置路径匹配器
	 */
	default void configurePathMatch(PathMatchConfigurer configurer) {
	}

	/**
	 * 配置内容协商主要的作用
	 * 1. 根据请求头来进行内容协商：例如，Accept: application/json 表示客户端希望接收 JSON 格式的数据
	 * 2. 根据文件扩展名进行内容协商：例如，/resource.json 表示客户端希望接收 JSON 格式的数据
	 * 3. 根据参数进行内容协商: 例如，/resource?format=json 表示客户端希望接收 JSON 格式的数据
	 * 4. 配置默认的内容类型，当无法从请求中确定内容类型时使用
	 * 5. 配置媒体类型解析器（Media Type Resolvers），用于将请求映射到具体的内容类型
	 */
	default void configureContentNegotiation(ContentNegotiationConfigurer configurer) {
	}

	/**
	 * 配置异步支持器
	 */
	default void configureAsyncSupport(AsyncSupportConfigurer configurer) {
	}

	/**
	 * 配置默认的Servlet
	 */
	default void configureDefaultServletHandling(DefaultServletHandlerConfigurer configurer) {
	}

	/**
	 * Add {@link Converter Converters} and {@link Formatter Formatters} in addition to the ones
	 * registered by default.
	 */
	default void addFormatters(FormatterRegistry registry) {
	}

	/**
	 * Add Spring MVC lifecycle interceptors for pre- and post-processing of
	 * controller method invocations and resource handler requests.
	 * Interceptors can be registered to apply to all requests or be limited
	 * to a subset of URL patterns.
	 */
	default void addInterceptors(InterceptorRegistry registry) {
	}

	/**
	 * Add handlers to serve static resources such as images, js, and, css
	 * files from specific locations under web application root, the classpath,
	 * and others.
	 * @see ResourceHandlerRegistry
	 */
	default void addResourceHandlers(ResourceHandlerRegistry registry) {
	}

	/**
	 * Configure "global" cross origin request processing. The configured CORS
	 * mappings apply to annotated controllers, functional endpoints, and static
	 * resources.
	 * <p>Annotated controllers can further declare more fine-grained config via
	 * {@link org.springframework.web.bind.annotation.CrossOrigin @CrossOrigin}.
	 * In such cases "global" CORS configuration declared here is
	 * {@link org.springframework.web.cors.CorsConfiguration#combine(CorsConfiguration) combined}
	 * with local CORS configuration defined on a controller method.
	 * @since 4.2
	 * @see CorsRegistry
	 * @see CorsConfiguration#combine(CorsConfiguration)
	 */
	default void addCorsMappings(CorsRegistry registry) {
	}

	/**
	 * Configure simple automated controllers pre-configured with the response
	 * status code and/or a view to render the response body. This is useful in
	 * cases where there is no need for custom controller logic -- e.g. render a
	 * home page, perform simple site URL redirects, return a 404 status with
	 * HTML content, a 204 with no content, and more.
	 * @see ViewControllerRegistry
	 */
	default void addViewControllers(ViewControllerRegistry registry) {
	}

	/**
	 * Configure view resolvers to translate String-based view names returned from
	 * controllers into concrete {@link org.springframework.web.servlet.View}
	 * implementations to perform rendering with.
	 * @since 4.1
	 */
	default void configureViewResolvers(ViewResolverRegistry registry) {
	}

	/**
	 * Add resolvers to support custom controller method argument types.
	 * <p>This does not override the built-in support for resolving handler
	 * method arguments. To customize the built-in support for argument
	 * resolution, configure {@link RequestMappingHandlerAdapter} directly.
	 * @param resolvers initially an empty list
	 */
	default void addArgumentResolvers(List<HandlerMethodArgumentResolver> resolvers) {
	}

	/**
	 * Add handlers to support custom controller method return value types.
	 * <p>Using this option does not override the built-in support for handling
	 * return values. To customize the built-in support for handling return
	 * values, configure RequestMappingHandlerAdapter directly.
	 * @param handlers initially an empty list
	 */
	default void addReturnValueHandlers(List<HandlerMethodReturnValueHandler> handlers) {
	}

	/**
	 * Configure the {@link HttpMessageConverter HttpMessageConverter}s for
	 * reading from the request body and for writing to the response body.
	 * <p>By default, all built-in converters are configured as long as the
	 * corresponding 3rd party libraries such Jackson JSON, JAXB2, and others
	 * are present on the classpath.
	 * <p><strong>Note</strong> use of this method turns off default converter
	 * registration. Alternatively, use
	 * {@link #extendMessageConverters(java.util.List)} to modify that default
	 * list of converters.
	 * @param converters initially an empty list of converters
	 */
	default void configureMessageConverters(List<HttpMessageConverter<?>> converters) {
	}

	/**
	 * Extend or modify the list of converters after it has been, either
	 * {@link #configureMessageConverters(List) configured} or initialized with
	 * a default list.
	 * <p>Note that the order of converter registration is important. Especially
	 * in cases where clients accept {@link org.springframework.http.MediaType#ALL}
	 * the converters configured earlier will be preferred.
	 * @param converters the list of configured converters to be extended
	 * @since 4.1.3
	 */
	default void extendMessageConverters(List<HttpMessageConverter<?>> converters) {
	}

	/**
	 * Configure exception resolvers.
	 * <p>The given list starts out empty. If it is left empty, the framework
	 * configures a default set of resolvers, see
	 * {@link WebMvcConfigurationSupport#addDefaultHandlerExceptionResolvers(List, org.springframework.web.accept.ContentNegotiationManager)}.
	 * Or if any exception resolvers are added to the list, then the application
	 * effectively takes over and must provide, fully initialized, exception
	 * resolvers.
	 * <p>Alternatively you can use
	 * {@link #extendHandlerExceptionResolvers(List)} which allows you to extend
	 * or modify the list of exception resolvers configured by default.
	 * @param resolvers initially an empty list
	 * @see #extendHandlerExceptionResolvers(List)
	 * @see WebMvcConfigurationSupport#addDefaultHandlerExceptionResolvers(List, org.springframework.web.accept.ContentNegotiationManager)
	 */
	default void configureHandlerExceptionResolvers(List<HandlerExceptionResolver> resolvers) {
	}

	/**
	 * Extending or modify the list of exception resolvers configured by default.
	 * This can be useful for inserting a custom exception resolver without
	 * interfering with default ones.
	 * @param resolvers the list of configured resolvers to extend
	 * @since 4.3
	 * @see WebMvcConfigurationSupport#addDefaultHandlerExceptionResolvers(List, org.springframework.web.accept.ContentNegotiationManager)
	 */
	default void extendHandlerExceptionResolvers(List<HandlerExceptionResolver> resolvers) {
	}

	/**
	 * Provide a custom {@link Validator} instead of the one created by default.
	 * The default implementation, assuming JSR-303 is on the classpath, is:
	 * {@link org.springframework.validation.beanvalidation.OptionalValidatorFactoryBean}.
	 * Leave the return value as {@code null} to keep the default.
	 */
	@Nullable
	default Validator getValidator() {
		return null;
	}

	/**
	 * Provide a custom {@link MessageCodesResolver} for building message codes
	 * from data binding and validation error codes. Leave the return value as
	 * {@code null} to keep the default.
	 */
	@Nullable
	default MessageCodesResolver getMessageCodesResolver() {
		return null;
	}
}
```



## 3. 初始化

### 3.1 RequestMappingHandlerMapping

请求映射器的处理器，用于解析 @RequestMapping 注解，将对应的controller中的方法解析成对应的 **HandlerMethod**

![image-20240428154241191](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20240428154241191.png)

### 3.2 AbstractHandlerMethodMapping

继承的核心方法主要是在 **AbstractHandlerMethodMapping** 中的 **afterPropertiesSet()** 方法中进行初始化处理，因为实现了 **InitializingBean** 接口，所以容器启动时就会进行初始化的解析

```java
@Override
public void afterPropertiesSet() {
  //调用初始化 handler方法，RequestMappingHandlerMapping子类将这个方法进行覆写了，只在里面配置相关配置信息，然后调用父类
  initHandlerMethods();
}
```

从spring容器中会获取到所有的bean对象

```java
protected void initHandlerMethods() {
		//获取到spring容器中所有的bean的名称
		for (String beanName : getCandidateBeanNames()) {
			//这里判断了一下bean的名称是否是 scopedTarget. 开头的，如果是的话，就不处理了，因为这个是spring内部使用的
			if (!beanName.startsWith(SCOPED_TARGET_NAME_PREFIX)) {
				//处理Bean对象
				processCandidateBean(beanName);
			}
		}
		handlerMethodsInitialized(getHandlerMethods());
	}
```

处理的方法主要的步骤是

- 先调用子类实现的方法 **isHandler()** 方法判断是否是需要处理的类，这里是 **RequestMappingHandlerMapping**进行实现的方法
- 将所有的方法中所有标识了 **@RequestMapping** 注解的方法全部解析为 **RequestMappingInfo** 类
- 最后将对应的关系，目标类、执行的方法、映射的配置类（RequestMappingInfo）注册到 **MappingRegistry** 中

```java
protected void processCandidateBean(String beanName) {
		Class<?> beanType = null;
		try {
			//通过spring容器获取到bean的类型
			beanType = obtainApplicationContext().getType(beanName);
		}
		catch (Throwable ex) {
			// An unresolvable bean type, probably from a lazy bean - let's ignore it.
			if (logger.isTraceEnabled()) {
				logger.trace("Could not resolve type for bean '" + beanName + "'", ex);
			}
		}
		//判断bean对象是否有 RequestMapping或者Controller注解
		if (beanType != null && isHandler(beanType)) {
			//处理 mvc请求类的方法
			detectHandlerMethods(beanName);
		}
	}
```

```java
protected void detectHandlerMethods(Object handler) {
		//获取到处理bean的类型
		Class<?> handlerType = (handler instanceof String ?
				obtainApplicationContext().getType((String) handler) : handler.getClass());

		if (handlerType != null) {
			Class<?> userType = ClassUtils.getUserClass(handlerType);
			//将所有的方法都解析为 RequestMappingInfo 类型
			Map<Method, T> methods = MethodIntrospector.selectMethods(userType,
					(MethodIntrospector.MetadataLookup<T>) method -> {
						try {
							//将method方法对象转换为 T 的泛型对象，是由 RequestMappingHandlerMapping实现所以T是RequestMappingInfo类型
							return getMappingForMethod(method, userType);
						}
						catch (Throwable ex) {
							throw new IllegalStateException("Invalid mapping on handler class [" +
									userType.getName() + "]: " + method, ex);
						}
					});
			if (logger.isTraceEnabled()) {
				logger.trace(formatMappings(userType, methods));
			}
			else if (mappingsLogger.isDebugEnabled()) {
				mappingsLogger.debug(formatMappings(userType, methods));
			}
			methods.forEach((method, mapping) -> {
				//将方法对象转换为可执行的方法对象
				Method invocableMethod = AopUtils.selectInvocableMethod(method, userType);
				//将bean和执行方法以及 RequestMappingInfo对象注册到mappingRegistry中，进行关联起来
				registerHandlerMethod(handler, invocableMethod, mapping);
			});
		}
	}
```

### 3.3 MappingRegistry

- 先将类和方法创建一个 **HandlerMethod** 关联类
- 存入到缓存中的是 **RequestMappingInfo** 作为key，value值创建一个 **MappingRegistration** 对象，里面保存了映射的路径、目标的对象等信息

```java
public void register(T mapping, Object handler, Method method) {
			this.readWriteLock.writeLock().lock();
			try {
				HandlerMethod handlerMethod = createHandlerMethod(handler, method);
				validateMethodMapping(handlerMethod, mapping);

				Set<String> directPaths = AbstractHandlerMethodMapping.this.getDirectPaths(mapping);
				for (String path : directPaths) {
					this.pathLookup.add(path, mapping);
				}

				String name = null;
				if (getNamingStrategy() != null) {
					name = getNamingStrategy().getName(handlerMethod, mapping);
					addMappingName(name, handlerMethod);
				}
				/**
				 * 处理 @CrossOrigin 注解，将注解构建成为 CorsConfiguration 配置对象
				 * 先获取到类上面的 @CrossOrigin 注解，在获取到方法上面的 @CrossOrigin 注解
				 */
				CorsConfiguration corsConfig = initCorsConfiguration(handler, method, mapping);
				if (corsConfig != null) {
					corsConfig.validateAllowCredentials();
					this.corsLookup.put(handlerMethod, corsConfig);
				}

				this.registry.put(mapping,
						new MappingRegistration<>(mapping, handlerMethod, directPaths, name, corsConfig != null));
			}
			finally {
				this.readWriteLock.writeLock().unlock();
			}
		}
```



### 3.4 DispatcherServlet

下面就是 **DispatcherServlet** 的继承图，从 **HttpServlet** 截至到上面部分都是 **Javax.servlet.http** 所提供的功能，而 **HttpServletBean** 开始就是 **spring** 实现的功能

![](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/DispatcherServlet.png)

DispatcherServlet比较核心的属性有以下：

- MultipartResolver：文件的解析器
- LocaleResolver：国际化的解析器
- ThemeResolver：主题解析器
- **HandlerMapping**：处理器映射器，从容器中获取出的 **List** 

#### 3.4.1 init()

在 tomcat 启动时并不会去对 **DispatcherServlet** 进行初始化，而是会在 **tomcat** 接收到请求时，如果 **Dispatcher** 还没有被初始化时，这时就会调用 **DispatcherServlet** 的 **init()** 方法进行初始化。初始化方法 **init()** 定义在 **GenericServlet** 中，交给子类进行实现，实际实现类是 **HttpServletBean** 类中对其进行实现

```java
public final void init() throws ServletException {

		//设置Servlet的配置属性
		PropertyValues pvs = new ServletConfigPropertyValues(getServletConfig(), this.requiredProperties);
		//判断配置属性是否是空的
		if (!pvs.isEmpty()) {
			try {
				//将当前HttpServletBean创建一个 BeanWrapperImpl包装实现类
				BeanWrapper bw = PropertyAccessorFactory.forBeanPropertyAccess(this);
				//获取到资源加载器
				ResourceLoader resourceLoader = new ServletContextResourceLoader(getServletContext());
				//注册自定义编辑器
				bw.registerCustomEditor(Resource.class, new ResourceEditor(resourceLoader, getEnvironment()));
				//初始化
				initBeanWrapper(bw);
				//注入配置信息
				bw.setPropertyValues(pvs, true);
			}
			catch (BeansException ex) {
				if (logger.isErrorEnabled()) {
					logger.error("Failed to set bean properties on servlet '" + getServletName() + "'", ex);
				}
				throw ex;
			}
		}
		//初始化DispatcherServlet，将所有请求的方法和适配器类初始化出来
		initServletBean();
	}
```

#### 3.4.2 initServletBean()

**initServletBean() 方法交给子类来进行实现**

```java
protected void initServletBean() throws ServletException {
}
```

实际实现的类是 **FrameworkServlet.initServletBean()** 进行初始化，然后调用 **initWebApplicationContext()** 继续交给子类进行实现

```java
protected final void initServletBean() throws ServletException {
		try {
			//初始化web应用的上下文
			this.webApplicationContext = initWebApplicationContext();
			initFrameworkServlet();
		}
		catch (ServletException | RuntimeException ex) {
			logger.error("Context initialization failed", ex);
			throw ex;
		}
	}
```

#### 3.4.3 initWebApplicationContext()

**initWebApplicationContext() ** 方法调用 **onRefresh()** 进行初始化

```java
protected WebApplicationContext initWebApplicationContext() {
		WebApplicationContext rootContext =
				WebApplicationContextUtils.getWebApplicationContext(getServletContext());
		WebApplicationContext wac = null;

		if (this.webApplicationContext != null) {
			// A context instance was injected at construction time -> use it
			wac = this.webApplicationContext;
			if (wac instanceof ConfigurableWebApplicationContext) {
				ConfigurableWebApplicationContext cwac = (ConfigurableWebApplicationContext) wac;
				if (!cwac.isActive()) {
					if (cwac.getParent() == null) {
						cwac.setParent(rootContext);
					}
					configureAndRefreshWebApplicationContext(cwac);
				}
			}
		}
		if (wac == null) {
			wac = findWebApplicationContext();
		}
		if (wac == null) {
			wac = createWebApplicationContext(rootContext);
		}

		if (!this.refreshEventReceived) {
			synchronized (this.onRefreshMonitor) {
        //进行初始化
				onRefresh(wac);
			}
		}
		if (this.publishContext) {
			String attrName = getServletContextAttributeName();
			getServletContext().setAttribute(attrName, wac);
		}
		return wac;
	}
```

#### 3.4.4 onRefresh()

**onRefresh** 方法交给子类进行实现，具体的子类是 **DispatcherServlet** 类，也是核心类

```java
protected void onRefresh(ApplicationContext context) {
  initStrategies(context);
}
```

#### 3.4.5 initStrategies()

```java
/**
	 * WebMvcConfigurationSupport：给容器中添加了很多的组件，
	 * 	例如：RequestMappingHandlerMapping 用于处理@RequestMapping 注解的类，RequestMappingHandlerAdapter处理器适配器，SimpleUrlHandlerMapping
	 * 	处理静态资源（通过WebMvcConfigurer进行注册静态资源的处理路径）
	 *
	 * @EnableWebMvc：注解导入了 DelegatingWebMvcConfiguration 它继承至 WebMvcConfigurationSupport，其中通过自动注入注入了所有的 WebMvcConfigurer 实现类
	 */
	protected void initStrategies(ApplicationContext context) {
		//初始化多部分文件解析器，请求头中是否有 multipart/form-data 属性，文件上传
		initMultipartResolver(context);
		//初始化本地化解析器
		initLocaleResolver(context);
		//初始化主题解析器
		initThemeResolver(context);
		/**
		 * 初始化处理器映射器，获取到容器中所有 HandlerMapping 实现类
		 * 配置时只需要配置 AbstractHandlerMethodMapping 的实现类就可以自动扫描出所有的 @RequestMapping 的类到容器中
		 * AbstractHandlerMethodMapping：实现了 InitializingBean 接口会调用 afterPropertiesSet() 去容器中获取到所有的bean对象，
		 * 								然后进行判断bean对象@Controller注解或者@RequestMapping注解，然后获取到其中的方法中含有
		 * 								注解@RequestMapping的方法，将其注册到内部的 MappingRegistry容器中
		 */
		initHandlerMappings(context);
		/**
		 * 初始化处理器是配置，接口类型是 HandlerAdapter 接口
		 * 使用的实现类：RequestMappingHandlerAdapter；其中会处理 @ControllerAdvice注解
		 */
		initHandlerAdapters(context);
		/**
		 * 从spring中获取到所有的异常处理器，然后将其注册到容器中
		 * HandlerExceptionResolver：
		 * 	ExceptionHandlerExceptionResolver.afterPropertiesSet() 中取读取到 @ControllerAdvice 注解的类
		 * 	然后通过ExceptionHandlerMethodResolver 处理@ExceptionHandler，进行异常方法的映射
		 * 	对于异常的执行来说是通过 DispatcherServlet.processDispatchResult() 中进行处理的
		 */
		initHandlerExceptionResolvers(context);
		//请求视图名称转换器
		initRequestToViewNameTranslator(context);
		//视图解析器
		initViewResolvers(context);
		//初始化缓存管理器
		initFlashMapManager(context);
	}
```

#### 2.4.6 initMultipartResolver

直接从容器中获取到对应类型的bean对象

```java
private void initMultipartResolver(ApplicationContext context) {
		try {
			this.multipartResolver = context.getBean(MULTIPART_RESOLVER_BEAN_NAME, MultipartResolver.class);
	}
```

#### 2.4.7 initLocaleResolver

```java
private void initLocaleResolver(ApplicationContext context) {
		try {
			this.localeResolver = context.getBean(LOCALE_RESOLVER_BEAN_NAME, LocaleResolver.class);
		}
		catch (NoSuchBeanDefinitionException ex) {
			// We need to use the default.
			this.localeResolver = getDefaultStrategy(context, LocaleResolver.class);
		}
	}
```

#### 2.4.8 initHandlerMappings

初始化构建处理器映射器，将 **RequestMappingHandlerMapping** 配置的类读取出来，请求来时通过处理器映射器来获取到对应的 **@RequestMapping** 标识的方法

```java
private void initHandlerMappings(ApplicationContext context) {
		this.handlerMappings = null;
		//判断是否需要构建所有的 HandlerMapping 接口的实现类，如果不需要构建所有的，只需要找到名称为 handlerMapping 的bean对象
		if (this.detectAllHandlerMappings) {
			// WebMvcConfigurationSupport 中配置了默认的请求处理器映射器以及资源处理器等包括 RequestMappingHandlerAdapter
			Map<String, HandlerMapping> matchingBeans =
					BeanFactoryUtils.beansOfTypeIncludingAncestors(context, HandlerMapping.class, true, false);
			if (!matchingBeans.isEmpty()) {
				this.handlerMappings = new ArrayList<>(matchingBeans.values());
				// We keep HandlerMappings in sorted order.
				AnnotationAwareOrderComparator.sort(this.handlerMappings);
			}
		}
		else {
			try {
				HandlerMapping hm = context.getBean(HANDLER_MAPPING_BEAN_NAME, HandlerMapping.class);
				this.handlerMappings = Collections.singletonList(hm);
			}
			catch (NoSuchBeanDefinitionException ex) {
				// Ignore, we'll add a default HandlerMapping later.
			}
		}
		if (this.handlerMappings == null) {
			this.handlerMappings = getDefaultStrategies(context, HandlerMapping.class);
			if (logger.isTraceEnabled()) {
				logger.trace("No HandlerMappings declared for servlet '" + getServletName() +
						"': using default strategies from DispatcherServlet.properties");
			}
		}

		for (HandlerMapping mapping : this.handlerMappings) {
			if (mapping.usesPathPatterns()) {
				this.parseRequestPath = true;
				break;
			}
		}
	}
```

#### 3.4.9 initHandlerAdapters

读取到对应的处理器适配器，通过适配器来执行对应的方法

```java
private void initHandlerAdapters(ApplicationContext context) {
		this.handlerAdapters = null;
		//判断是否需要加载出所有的spring中的 HandlerAdapter 实现类，如果不需要加入的话只会加载默认的名称叫 handlerAdapter 的适配器
		if (this.detectAllHandlerAdapters) {
			//通过spring容器加载出所有的 HandlerAdapter 实现类
			Map<String, HandlerAdapter> matchingBeans =
					BeanFactoryUtils.beansOfTypeIncludingAncestors(context, HandlerAdapter.class, true, false);
			if (!matchingBeans.isEmpty()) {
				this.handlerAdapters = new ArrayList<>(matchingBeans.values());
				//根据@Order进行排序
				AnnotationAwareOrderComparator.sort(this.handlerAdapters);
			}
		}
		else {
			try {
				HandlerAdapter ha = context.getBean(HANDLER_ADAPTER_BEAN_NAME, HandlerAdapter.class);
				this.handlerAdapters = Collections.singletonList(ha);
			}
			catch (NoSuchBeanDefinitionException ex) {
				// Ignore, we'll add a default HandlerAdapter later.
			}
		}
		if (this.handlerAdapters == null) {
			this.handlerAdapters = getDefaultStrategies(context, HandlerAdapter.class);
			if (logger.isTraceEnabled()) {
				logger.trace("No HandlerAdapters declared for servlet '" + getServletName() +
						"': using default strategies from DispatcherServlet.properties");
			}
		}
	}
```

#### 2.4.10 initHandlerExceptionResolvers

初始化异常相关的处理类

```java
private void initHandlerExceptionResolvers(ApplicationContext context) {
		this.handlerExceptionResolvers = null;

		if (this.detectAllHandlerExceptionResolvers) {
			// Find all HandlerExceptionResolvers in the ApplicationContext, including ancestor contexts.
			Map<String, HandlerExceptionResolver> matchingBeans = BeanFactoryUtils
					.beansOfTypeIncludingAncestors(context, HandlerExceptionResolver.class, true, false);
			if (!matchingBeans.isEmpty()) {
				this.handlerExceptionResolvers = new ArrayList<>(matchingBeans.values());
				// We keep HandlerExceptionResolvers in sorted order.
				AnnotationAwareOrderComparator.sort(this.handlerExceptionResolvers);
			}
		}
		else {
			try {
				HandlerExceptionResolver her =
						context.getBean(HANDLER_EXCEPTION_RESOLVER_BEAN_NAME, HandlerExceptionResolver.class);
				this.handlerExceptionResolvers = Collections.singletonList(her);
			}
			catch (NoSuchBeanDefinitionException ex) {
				// Ignore, no HandlerExceptionResolver is fine too.
			}
		}
		if (this.handlerExceptionResolvers == null) {
			this.handlerExceptionResolvers = getDefaultStrategies(context, HandlerExceptionResolver.class);
			if (logger.isTraceEnabled()) {
				logger.trace("No HandlerExceptionResolvers declared in servlet '" + getServletName() +
						"': using default strategies from DispatcherServlet.properties");
			}
		}
	}
```



## 4. 请求处理

### 4.1 DispatcherServlet

#### 4.1.1 doDispatch()

真正处理请求的方法

```java
protected void doDispatch(HttpServletRequest request, HttpServletResponse response) throws Exception {
		HttpServletRequest processedRequest = request;
		//处理器执行链
		HandlerExecutionChain mappedHandler = null;
		//是否多部分请求解析（文件上传）
		boolean multipartRequestParsed = false;
		//向http请求中的 Attribute 属性中设置一个异步管理器
		WebAsyncManager asyncManager = WebAsyncUtils.getAsyncManager(request);

		try {
			//模型和视图对象
			ModelAndView mv = null;
			//捕获的异常
			Exception dispatchException = null;

			try {
				/**
				 * 检查http请求是否是文件上传，如果是文件上传的话将其包装为 DefaultMultipartHttpServletRequest 并且对文件对象进行解析
				 */
				processedRequest = checkMultipart(request);
				multipartRequestParsed = (processedRequest != request);

				/**
				 * 根据请求的路径获取到对应的handler处理器
				 * HandlerExecutionChain：执行的请求链类，最终使用的是里面的 handler
				 *
				 * AbstractHandlerMethodMapping：这个抽象类是所有的处理器映射器的父类，在配置到容器中后通过 afterPropertiesSet() 方法会将所有的方法进行初始化
				 * 而具体的实现类则是 RequestMappingHandlerMapping
				 */
				mappedHandler = getHandler(processedRequest);
				if (mappedHandler == null) {
					//处理没有找到对应的处理器映射器
					noHandlerFound(processedRequest, response);
					return;
				}

				/**
				 * 根据处理器找到对应的处理器适配器
				 * 返回的是 RequestMappingHandlerAdapter 适配器进行处理
				 */
				HandlerAdapter ha = getHandlerAdapter(mappedHandler.getHandler());

				// Process last-modified header, if supported by the handler.
				String method = request.getMethod();
				boolean isGet = HttpMethod.GET.matches(method);
				if (isGet || HttpMethod.HEAD.matches(method)) {
					long lastModified = ha.getLastModified(request, mappedHandler.getHandler());
					if (new ServletWebRequest(request, response).checkNotModified(lastModified) && isGet) {
						return;
					}
				}
				/**
				 * 获取到拦截器进行执行
				 */
				if (!mappedHandler.applyPreHandle(processedRequest, response)) {
					return;
				}

				/**
				 * 真正执行请求处理的适配器
				 * RequestMappingHandlerAdapter，最终调用到的父类方法是 AbstractHandlerMethodAdapter.handle()
				 * 其中通过从处理器映射器获取到的handler进行处理
				 * 说明：
				 * RequestMappingHandlerAdapter中初始了很多的处理，例如：http消息的转换、返回值的处理等，在 afterPropertiesSet() 中初始化了很多的处理器
				 */
				mv = ha.handle(processedRequest, response, mappedHandler.getHandler());

				if (asyncManager.isConcurrentHandlingStarted()) {
					return;
				}
				//根据请求设置默认的视图名称
				applyDefaultViewName(processedRequest, mv);
				//执行拦截器的后置处理
				mappedHandler.applyPostHandle(processedRequest, response, mv);
			}
			catch (Exception ex) {
				dispatchException = ex;
			}
			catch (Throwable err) {
				// As of 4.3, we're processing Errors thrown from handler methods as well,
				// making them available for @ExceptionHandler methods and other scenarios.
				dispatchException = new NestedServletException("Handler dispatch failed", err);
			}
			//处理请求结果
			processDispatchResult(processedRequest, response, mappedHandler, mv, dispatchException);
		}
		catch (Exception ex) {
			//触发拦截器的完成处理
			triggerAfterCompletion(processedRequest, response, mappedHandler, ex);
		}
		catch (Throwable err) {
			triggerAfterCompletion(processedRequest, response, mappedHandler,
					new NestedServletException("Handler processing failed", err));
		}
		finally {
			if (asyncManager.isConcurrentHandlingStarted()) {
				// Instead of postHandle and afterCompletion
				if (mappedHandler != null) {
					mappedHandler.applyAfterConcurrentHandlingStarted(processedRequest, response);
				}
			}
			else {
				// Clean up any resources used by a multipart request.
				if (multipartRequestParsed) {
					cleanupMultipart(processedRequest);
				}
			}
		}
	}
```

### 4.2 RequestMappingHandlerAdapter

最终 **doDispatch()** 方法调用的类就是 **RequestMappingHandlerAdapter.invokeHandlerMethod()** 方法

```java
protected ModelAndView invokeHandlerMethod(HttpServletRequest request,
			HttpServletResponse response, HandlerMethod handlerMethod) throws Exception {
		//包装请求对象
		ServletWebRequest webRequest = new ServletWebRequest(request, response);
		try {
			/**
			 * ServletRequestDataBinderFactory：构建一个参数绑定的工厂，会将当前mvc bean对象中 有 @InitBinder 注解的方法读取出来
			 * 再对请求接口转换时用于参数的转换
			 */
			WebDataBinderFactory binderFactory = getDataBinderFactory(handlerMethod);
			/**
			 * 构建 ModelFactory工厂，读取出所有的 @ModelAttribute 方法
			 */
			ModelFactory modelFactory = getModelFactory(handlerMethod, binderFactory);

			//包装一下请求的方法对象
			ServletInvocableHandlerMethod invocableMethod = createInvocableHandlerMethod(handlerMethod);
			//参数解析器 HandlerMethodArgumentResolverComposite
			if (this.argumentResolvers != null) {
				invocableMethod.setHandlerMethodArgumentResolvers(this.argumentResolvers);
			}
			//返回值处理器 HandlerMethodReturnValueHandlerComposite
			if (this.returnValueHandlers != null) {
				invocableMethod.setHandlerMethodReturnValueHandlers(this.returnValueHandlers);
			}
			invocableMethod.setDataBinderFactory(binderFactory);
			invocableMethod.setParameterNameDiscoverer(this.parameterNameDiscoverer);

			//模型视图容器
			ModelAndViewContainer mavContainer = new ModelAndViewContainer();
			mavContainer.addAllAttributes(RequestContextUtils.getInputFlashMap(request));
			modelFactory.initModel(webRequest, mavContainer, invocableMethod);
			mavContainer.setIgnoreDefaultModelOnRedirect(this.ignoreDefaultModelOnRedirect);
			//StandardServletAsyncWebRequest 创建一个web的异步请求
			AsyncWebRequest asyncWebRequest = WebAsyncUtils.createAsyncWebRequest(request, response);
			asyncWebRequest.setTimeout(this.asyncRequestTimeout);

			WebAsyncManager asyncManager = WebAsyncUtils.getAsyncManager(request);
			asyncManager.setTaskExecutor(this.taskExecutor);
			asyncManager.setAsyncWebRequest(asyncWebRequest);
			//注册拦截器
			asyncManager.registerCallableInterceptors(this.callableInterceptors);
			//结果拦截器
			asyncManager.registerDeferredResultInterceptors(this.deferredResultInterceptors);

			if (asyncManager.hasConcurrentResult()) {
				Object result = asyncManager.getConcurrentResult();
				mavContainer = (ModelAndViewContainer) asyncManager.getConcurrentResultContext()[0];
				asyncManager.clearConcurrentResult();
				LogFormatUtils.traceDebug(logger, traceOn -> {
					String formatted = LogFormatUtils.formatValue(result, !traceOn);
					return "Resume with async result [" + formatted + "]";
				});
				invocableMethod = invocableMethod.wrapConcurrentResult(result);
			}
			//执行方法
			invocableMethod.invokeAndHandle(webRequest, mavContainer);
			if (asyncManager.isConcurrentHandlingStarted()) {
				return null;
			}

			return getModelAndView(mavContainer, modelFactory, webRequest);
		}
		finally {
			webRequest.requestCompleted();
		}
	}
```



