---
title: SpringBoot实现多数据源切换
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
thumbnail: https://cdn2.zzzmh.cn/wallpaper/origin/42848fd7880e11ebb6edd017c2d2eca2.jpg/fhd?auth_key=1749052800-a9d6db4059f59d93c3c9dff4b20c7e6a2df84469-0-65716973d8e5dd156f418d9e16581d34
published: true
---
# springboot实现多数据源切换

## 1. 实现效果

### 1.1 controller

最终实现效果，在接口上标记上 **@Router** 注解用来标记当前接口需要根据参数中的某个字段进行数据的切换

```java
@RestController
@RequestMapping("/user")
public class UserController {

    @Resource
    private UserMapper userMapper;

    @GetMapping("/save")
    @Router(routingFiled = "id")
    @Transactional
    public void saveUser(@RequestParam("id") String id){
        User user = new User();
        user.setAge("123");
        user.setId(Long.valueOf(id));
        user.setName("zs");
        //设置表的后缀名称，在mybatis执行sql时可以获取
        user.setTableSuffix(MultiDataSourceHolder.getTableIndex());
        userMapper.insert(user);
    }

    @GetMapping("/get")
    @Router(routingFiled = "id")
    public User getUser(@RequestParam("id") String id){
        return userMapper.selectByPrimaryKey(Long.valueOf(id),MultiDataSourceHolder.getTableIndex());
    }
}
```

### 1.2 mybatis.xml

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE mapper PUBLIC "-//mybatis.org//DTD Mapper 3.0//EN" "http://mybatis.org/dtd/mybatis-3-mapper.dtd">
<mapper namespace="com.zhj.multiple.mapper.UserMapper">

  <resultMap id="BaseResultMap" type="com.zhj.multiple.entity.User">
    <!--@mbg.generated-->
    <id column="id" jdbcType="BIGINT" property="id" />
    <result column="name" jdbcType="VARCHAR" property="name" />
    <result column="age" jdbcType="VARCHAR" property="age" />
  </resultMap>

  <sql id="Base_Column_List">
    <!--@mbg.generated-->
    id, `name`, age
  </sql>

  <select id="selectByPrimaryKey" parameterType="java.lang.Long" resultMap="BaseResultMap">
    <!--@mbg.generated-->
    select 
    <include refid="Base_Column_List" />
      <!--通过${}获取出表名，为什么不是用#呢？因为表名是在内部进行计算，不用担心sql注入的问题，而且$进行获取是直接将表名进行拼接上，不会使用预处理-->
    from user${tableName}
    where id = #{id,jdbcType=BIGINT}
  </select>


  <insert id="insert" keyColumn="id" keyProperty="id" parameterType="com.zhj.multiple.entity.User" useGeneratedKeys="true">
    <!--@mbg.generated-->
    insert into user${tableSuffix} (id ,`name`, age)
    values (#{id},#{name,jdbcType=VARCHAR}, #{age,jdbcType=VARCHAR})
  </insert>

</mapper>
```

### 1.3 application.yml

```yml
#showSql
logging:
  level:
    com:
      zhj:
        mapper : debug
server:
  port: 8081
# 配置路由分库分表的策略
datasource:
  stragegy:
    dataSourceNum: 2         #库的个数
    tableNum: 2              #表的个数
    routingFiled: 'userId'     #根据哪个字段来进行分库分表
    tableSuffixStyle: '%04d'   #表的索引值 4位补齐  例如：_0003
    tableSuffixConnect: '_'  #表的连接风格 order_
    routingStategy: 'ROUTING_DS_TABLE_STATEGY'  #表的策略，启动时会根据表的策略进行验证
  #配置数据源
  multiple:
    data0:
      user: root
      password: root
      url: jdbc:mysql://192.168.60.46:3306/multiple-0?useUnicode=true&characterEncoding=utf-8&useSSL=true&serverTimezone=UTC
      driver: com.mysql.jdbc.Driver
    data1:
      user: root
      password: root
      url: jdbc:mysql://192.168.60.46:3306/multiple-1?useUnicode=true&characterEncoding=utf-8&useSSL=true&serverTimezone=UTC
      driver: com.mysql.jdbc.Driver
```



### 1.4 启动类

```java
@SpringBootApplication
@EnableAspectJAutoProxy
@MapperScan("com.zhj.multiple.mapper")
@EnableTransactionManagement
public class MultipleApplication {

    public static void main(String[] args) {
        SpringApplication.run(MultipleApplication.class, args);
    }

}
```



## 2. 注解

### 2.1 @Router

```java
/**
 * 路由注解
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.METHOD)
@Documented
public @interface Router {

    /**
     * 路由字段
     * 
     * @return 默认路由字段是参数中的哪一个
     */
    String routingFiled() default MultipleConstant.DEFAULT_ROUTING_FIELD;
}
```

## 3. 分库策略

### 3.1 MultipleConstant

目前有三种分库的策略：

- 多库多表
- 多库单表
- 单库多表

```java
@Data
public class MultipleConstant {
    /**
     * 多库多表策略
     */
    public static final String ROUTING_DS_TABLE_STATEGY = "ROUTING_DS_TABLE_STATEGY";

    /**
     * 多库单表策略
     */
    public static final String ROUTGING_DS_STATEGY = "ROUTGING_DS_STATEGY";

    /**
     * 单库多表策略
     */
    public static final String ROUTGIN_TABLE_STATEGY = "ROUTGIN_TABLE_STATEGY";

    /**
     * 默认的路由字段
     */
    public static final String DEFAULT_ROUTING_FIELD = "accountId";
}
```

### 3.2 IRoutingInterface

路由的顶级接口，用于定义一些通用的方法

```java
public interface IRoutingInterface {

    /**
     * 根据字段key计算出数据库
     * @param routingFiled
     * @return
     */
    String calDataSourceKey(String routingFiled);

    /**
     * 获取路由字段的hashCode
     * @param routingFiled
     * @return
     */
    Integer getRoutingFileHashCode(String routingFiled);

    /**
     * 计算出表名
     * @param routingFiled
     * @return
     */
    String calTableKey(String routingFiled);

    /**
     * 计算出表的前缀
     * @param tableIndex
     * @return
     */
    String getFormatTableSuffix(Integer tableIndex);

}
```

### 3.3 AbstractRouting

```java
@EnableConfigurationProperties({MultipleStategyProperties.class})
@Data
@Slf4j
public abstract class AbstractRouting implements IRoutingInterface, InitializingBean {

    @Resource
    private MultipleStategyProperties multipleStategyProperties;

    @Override
    public Integer getRoutingFileHashCode(String routingFiled){
        return Math.abs(routingFiled.hashCode());
    }

    /**
     * 获取表的后缀
     * @param tableIndex 表索引
     */
    @Override
    public String getFormatTableSuffix(Integer tableIndex){
        //获取连接符
        String tableSuffixConnect = multipleStategyProperties.getTableSuffixConnect();
        //根据配置风格格式化表后缀名称
        String format = String.format(multipleStategyProperties.getTableSuffixStyle(), tableIndex);
        return tableSuffixConnect + format;
    }


    /**
     * 工程启动时，检验配置的数据源是否跟策略相似，实现了 InitializingBean 初始化后会执行当前方法
     */
    @Override
    public void afterPropertiesSet(){
        switch (multipleStategyProperties.getRoutingStategy()) {
            case MultipleConstant.ROUTING_DS_TABLE_STATEGY:
                checkRoutingDsTableStategyConfig();
                break;
            case MultipleConstant.ROUTGING_DS_STATEGY:
                checkRoutingDsStategyConfig();
                break;
            default:
                checkRoutingTableStategyConfig();
                break;
        }
    }

    /**
     * 检查多库 多表配置
     */
    private void checkRoutingDsTableStategyConfig() {
        if(multipleStategyProperties.getTableNum()<=1 ||multipleStategyProperties.getDataSourceNum()<=1){
            log.error("你的配置项routingStategy:{}是多库多表配置,数据库个数>1," +
                            "每一个库中表的个数必须>1,您的配置:数据库个数:{},表的个数:{}",multipleStategyProperties.getRoutingStategy(),
                    multipleStategyProperties.getDataSourceNum(),multipleStategyProperties.getTableNum());
            throw new RuntimeException();
        }
    }

    /**
     * 检查多库一表的路由配置项
     */
    private void checkRoutingDsStategyConfig() {
        if(multipleStategyProperties.getTableNum()!=1 ||multipleStategyProperties.getDataSourceNum()<=1){
            log.error("你的配置项routingStategy:{}是多库一表配置,数据库个数>1," +
                            "每一个库中表的个数必须=1,您的配置:数据库个数:{},表的个数:{}",multipleStategyProperties.getRoutingStategy(),
                    multipleStategyProperties.getDataSourceNum(),multipleStategyProperties.getTableNum());
            throw new RuntimeException();
        }
    }

    /**
     * 检查一库多表的路由配置项
     */
    private void checkRoutingTableStategyConfig() {
        if(multipleStategyProperties.getTableNum()<=1 ||multipleStategyProperties.getDataSourceNum()!=1){
            log.error("你的配置项routingStategy:{}是一库多表配置,数据库个数=1," +
                            "每一个库中表的个数必须>1,您的配置:数据库个数:{},表的个数:{}",multipleStategyProperties.getRoutingStategy(),
                    multipleStategyProperties.getDataSourceNum(),multipleStategyProperties.getTableNum());
            throw new RuntimeException();
        }
    }

}
```

### 3.4 RoutingDsAndTbStrategy

目前实现了一个多库多表的策略进行配置，其余两个分库算法可以自行实现

```java
@Slf4j
public class RoutingDsAndTbStrategy extends AbstractRouting {

    /**
     * 确定数据源的key
     * @param routingFiled
     * @return
     */
    @Override
    public String calDataSourceKey(String routingFiled) {
        //计算hash值
        Integer routingFileHashCode = getRoutingFileHashCode(routingFiled);

        //定位数据源
        int dsIndex = routingFileHashCode % getMultipleStategyProperties().getDataSourceNum();

        String dataSourceKey = getMultipleStategyProperties().getDataSourceKeysMapping().get(dsIndex);

        //将数据源key放入持有器当中
        MultiDataSourceHolder.setDataSourceHolder(dataSourceKey);

        log.info("根据路由字段:{},值:{},计算出数据库索引值:{},数据源key的值:{}",getMultipleStategyProperties().getRoutingFiled(),routingFiled,dsIndex,dataSourceKey);

        return dataSourceKey;
    }

    /**
     * 计算表的key
     * @param routingFiled
     * @return
     */
    @Override
    public String calTableKey(String routingFiled) {
        //获取到当前key的hash
        Integer routingFileHashCode = getRoutingFileHashCode(routingFiled);
        //通过hash值取模，获取到对应的索引值
        int tbIndex = routingFileHashCode % getMultipleStategyProperties().getTableNum();

        //获取表后缀
        String formatTableSuffix = getFormatTableSuffix(tbIndex);
        //将表名设置到上下文中，方便后续同线程获取到对应的表名
        MultiDataSourceHolder.setTableIndexHolder(formatTableSuffix);

        return formatTableSuffix;
    }
}
```

### 3.5 RoutingDsStrategy

多库单表

```java
public class RoutingDsStrategy extends AbstractRouting {
    @Override
    public String calDataSourceKey(String routingFiled) {
        return null;
    }

    @Override
    public String calTableKey(String routingFiled) {
        return null;
    }
}
```

### 3.6 RoutingTbStrategy

单库多表策略

```java
public class RoutingTbStrategy extends AbstractRouting {
    @Override
    public String calDataSourceKey(String routingFiled) {
        return null;
    }

    @Override
    public String calTableKey(String routingFiled) {
        return null;
    }
}
```



## 4. 配置类

以下两个配置：

- MultipleStategyProperties：用于配置数据库策略，有多少库，多少表，以及表名

### 4.1 MultipleStategyProperties

```java
@ConfigurationProperties(prefix = "datasource.stragegy")
@Data
public class MultipleStategyProperties {

    /**
     * 默认是一个数据库 默认一个
     */
    private Integer dataSourceNum = 1;

    /**
     * 每一个库对应表的个数 默认是一个
     */
    private Integer tableNum = 1;

    /**
     * 路由字段 必须在配置文件中配置(不配置会抛出异常)
     */
    private String routingFiled;

    /**
     * 数据库的映射关系
     */
    private Map<Integer,String> dataSourceKeysMapping;

    /**
     * 表的后缀连接风格 比如order_
     */
    private String tableSuffixConnect="_";

    /**
     * 表的索引值 格式化为四位 不足左补零   1->0001 然后在根据tableSuffixConnect属性拼接成
     * 成一个完整的表名  比如 order表 所以为1  那么数据库表明为 order_0001
     */
    private String tableSuffixStyle= "%04d";


    /**
     * 默认的多库多表策略
     */
    private String routingStategy = MultipleConstant.ROUTING_DS_TABLE_STATEGY;


}
```

### 4.2 MultipleStategyProperties

```java
@ConfigurationProperties(prefix = "datasource.multiple")
@Data
public class MultipleDataSourceProperties {

    /** Data 0 */
    private DataSource data0;
    /** Data 1 */
    private DataSource data1;


    /**
     * 数据源配置
     */
    @Data
    public static class DataSource{
        /** User */
        private String user;
        /** Password */
        private String password;
        /** Url */
        private String url;
        /** Driver */
        private String driver;
    }

}
```



### 4.3 MultipleDataSourceStrategyConfig

根据对应的配置创建不同的分库策略

```java
@Configuration
public class MultipleDataSourceStrategyConfig {

    /**
     * 当配置文件里面包含某个配置，并且值是多少时生效
     *
     * @return routing interface
     * @since 1.0.0
     */
    @Bean
    @ConditionalOnProperty(prefix = "datasource.stragegy",name = "routingStategy",havingValue = "ROUTING_DS_TABLE_STATEGY")
    public IRoutingInterface routingDsAndTbStrategy(){
        return new RoutingDsAndTbStrategy();
    }

    /**
     * Routing ds strategy
     *
     * @return the routing interface
     * @since 1.0.0
     */
    @Bean
    @ConditionalOnProperty(prefix = "datasource.stragegy",name = "routingStategy",havingValue = "ROUTGING_DS_STATEGY")
    public IRoutingInterface routingDsStrategy(){
        return new RoutingDsStrategy();
    }

    /**
     * Routing tb strategy
     *
     * @return the routing interface
     * @since 1.0.0
     */
    @Bean
    @ConditionalOnProperty(prefix = "datasource.stragegy",name = "routingStategy",havingValue = "ROUTGIN_TABLE_STATEGY")
    public IRoutingInterface routingTbStrategy(){
        return new RoutingTbStrategy();
    }


}
```



### 4.4 MultipleDataSourceConfig

多数据源自动装配类，其中创建了多个数据源，通过 spring提供的 **AbstractRoutingDataSource** 类进行数据源的切换

```java
@Configuration
//开启数据源以及数据分库策略配置
@EnableConfigurationProperties({MultipleDataSourceProperties.class, MultipleStategyProperties.class})
public class MultipleDataSourceConfig {

    @Resource
    private MultipleDataSourceProperties multipleDataSourceProperties;

    @Resource
    private MultipleStategyProperties multipleStategyProperties;

    /**
     * 配置数据源
     * @return
     */
    @Bean("data0")
    public DataSource dataSource0(){
        DruidDataSource druidDataSource = new DruidDataSource();
        druidDataSource.setUsername(multipleDataSourceProperties.getData0().getUser());
        druidDataSource.setPassword(multipleDataSourceProperties.getData0().getPassword());
        druidDataSource.setUrl(multipleDataSourceProperties.getData0().getUrl());
        druidDataSource.setDriverClassName(multipleDataSourceProperties.getData0().getDriver());
        return druidDataSource;
    }
    @Bean("data1")
    public DataSource dataSource1(){
        DruidDataSource druidDataSource = new DruidDataSource();
        druidDataSource.setUsername(multipleDataSourceProperties.getData1().getUser());
        druidDataSource.setPassword(multipleDataSourceProperties.getData1().getPassword());
        druidDataSource.setUrl(multipleDataSourceProperties.getData1().getUrl());
        druidDataSource.setDriverClassName(multipleDataSourceProperties.getData1().getDriver());
        return druidDataSource;
    }

    /**
     * 设置多数据源
     * @param data0
     * @param data1
     * @return
     */
    @Bean
    public MultiDataSource multiDataSource(DataSource data0,DataSource data1){
        //将多个数据与数据源关联起来
        MultiDataSource multiDataSource = new MultiDataSource();
        HashMap<Object, Object> multiMap = new HashMap<>();
        multiMap.put("data0",data0);
        multiMap.put("data1",data1);
        //设置目标数据源
        multiDataSource.setTargetDataSources(multiMap);
        //设置默认的数据源
        multiDataSource.setDefaultTargetDataSource(data0);
        //设置数据源名称的映射
        Map<Integer, String> multiMappings = new HashMap<>();
        multiMappings.put(0,"data0");
        multiMappings.put(1,"data1");
        multipleStategyProperties.setDataSourceKeysMapping(multiMappings);

        return multiDataSource;
    }

    /**
     * 将多数据源设置进mybatis的工厂类中
     * @param multiDataSource
     * @return
     */
    @Bean
    public SqlSessionFactory sqlSessionFactory(@Qualifier("multiDataSource") MultiDataSource multiDataSource) throws Exception {
        SqlSessionFactoryBean sqlSessionFactoryBean = new SqlSessionFactoryBean();
        sqlSessionFactoryBean.setDataSource(multiDataSource);
        sqlSessionFactoryBean.setMapperLocations(new PathMatchingResourcePatternResolver().getResources("classpath:/mybatis/mappers/*.xml"));
        sqlSessionFactoryBean.setTypeAliasesPackage("com.zhj.multiple.entity");
        return sqlSessionFactoryBean.getObject();
    }

    @Bean
    public SqlSessionTemplate sqlSessionTemplate(SqlSessionFactory sqlSessionFactory){
        return new SqlSessionTemplate(sqlSessionFactory);
    }


    /**
     * 将多数据源设置到事务管理器中
     * 
     * @param multiDataSource
     * @return
     */
    @Bean
    public DataSourceTransactionManager dataSourceTransactionManager(@Qualifier("multiDataSource") MultiDataSource multiDataSource){
        return new DataSourceTransactionManager(multiDataSource);
    }

}
```

### 4.5 MultiDataSource

覆写 **AbstractRoutingDataSource.determineCurrentLookupKey()** 的方法，在mybatis中通过 **Datasource.getConnection()** 会调用 **determineCurrentLookupKey()** 获取到对应的数据源，然后通过数据源获取到连接，其中内部维护了一个 **Map** 来保存数据源的映射关系

```java
public class MultiDataSource extends AbstractRoutingDataSource {

    /**
     * 获取到指定的数据源
     * @return
     */
    @Override
    protected Object determineCurrentLookupKey() {
        return MultiDataSourceHolder.getDataSourceKey();
    }
}
```





## 5. 全局上下文

用于保存数据库、表名，方便后续使用

### 5.1 MultiDataSourceHolder

```java
@Data
public class MultiDataSourceHolder {

    /**
     * 存储数据源
     */
    private static final ThreadLocal<String> dataSourceHolder = new ThreadLocal<>();

    /**
     * 存储表的索引
     */
    private static final ThreadLocal<String> tableIndexHolder = new ThreadLocal<>();

    /**
     * Get data source key
     *
     * @return the string
     * @since 1.0.0
     */
    public static String getDataSourceKey(){
        return dataSourceHolder.get();
    }

    /**
     * Get table index
     *
     * @return the string
     * @since 1.0.0
     */
    public static String getTableIndex(){
        return tableIndexHolder.get();
    }

    /**
     * Clear data source key
     *
     * @since 1.0.0
     */
    public static void clearDataSourceKey(){
        dataSourceHolder.remove();
    }

    /**
     * Clear table index
     *
     * @since 1.0.0
     */
    public static void clearTableIndex(){
        tableIndexHolder.remove();
    }

    /**
     * Set data source holder
     *
     * @param key key
     * @since 1.0.0
     */
    public static void setDataSourceHolder(String key){
        dataSourceHolder.set(key);
    }

    /**
     * Set table index holder
     *
     * @param key key
     * @since 1.0.0
     */
    public static void setTableIndexHolder(String key){
        tableIndexHolder.set(key);
    }


}
```



## 6. 切面

### 6.1 RoutingAspect

通过路由切面进行 **@Router** 注解的处理，提前将数据库的key以及表名的后缀获取出来进行存储

```java
@Aspect
@Component
@Slf4j
public class RoutingAspect {

    @Resource
    private IRoutingInterface iRoutingInterface;
    
    @Pointcut("@annotation(com.zhj.multiple.annotations.Router)")
    public void pointCut(){};

    @Before("pointCut()")
    public void before(JoinPoint joinPoint) throws IllegalAccessException, NoSuchMethodException, InvocationTargetException {
        long beginTime = System.currentTimeMillis();
        //获取方法调用名称
        Method method = getInvokeMethod(joinPoint);
        //获取方法指定的注解
        Router router = method.getAnnotation(Router.class);
        //获取指定的路由key
        String routingFiled = router.routingFiled();

        if(Objects.nonNull(router)){
            boolean havingRoutingField = false;
            //获取到http请求
            HttpServletRequest requestAttributes = ((ServletRequestAttributes)RequestContextHolder.getRequestAttributes()).getRequest();
            //优先获取@ReqeustParam注解中的路由字段
            String routingFieldValue = requestAttributes.getParameter(routingFiled);
            if(!StringUtils.isEmpty(routingFieldValue)){
                //计算数据库key
                String dbKey = iRoutingInterface.calDataSourceKey(routingFieldValue);
                //计算表索引
                String tableIndex = iRoutingInterface.calTableKey(routingFieldValue);
                log.info("选择的dbkey是:{},tableKey是:{}",dbKey,tableIndex);
            }else {
                //获取方法入参
                Object[] args = joinPoint.getArgs();
                if(args != null && args.length > 0) {
                    for(int index = 0; index < args.length; index++) {
                        //找到参数当中路由字段的值
                        routingFieldValue = BeanUtils.getProperty(args[index],routingFiled);
                        if(!StringUtils.isEmpty(routingFieldValue)) {
                            //计算数据库key
                            String dbKey = iRoutingInterface.calDataSourceKey(routingFieldValue);
                            //计算表索引
                            String tableIndex = iRoutingInterface.calTableKey(routingFieldValue);
                            log.info("选择的dbkey是:{},tableKey是:{}",dbKey,tableIndex);
                            havingRoutingField = true;
                            break;
                        }
                    }
                    //判断入参中没有路由字段
                    if(!havingRoutingField) {
                        log.warn("入参{}中没有包含路由字段:{}",args,routingFiled);
                        throw new RuntimeException();
                    }
                }
            }
        }


    }


    private Method getInvokeMethod(JoinPoint joinPoint) {
        Signature signature = joinPoint.getSignature();
        MethodSignature methodSignature = (MethodSignature)signature;
        Method targetMethod = methodSignature.getMethod();
        return targetMethod;
    }

    /**
     * 清除线程缓存
     * @param joinPoint
     */
    @After("pointCut()")
    public void methodAfter(JoinPoint joinPoint){
        MultiDataSourceHolder.clearDataSourceKey();
        MultiDataSourceHolder.clearTableIndex();
    }


}
```

