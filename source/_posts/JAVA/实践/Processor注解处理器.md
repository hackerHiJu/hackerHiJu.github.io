---
title: Processor注解处理器
date: 2024-12-30 19:07:05
updated: 2024-12-30 19:07:05
tags:
  - Java
  - 源码
comments: true
categories:
  - Java
  - 源码
  - 集合
thumbnail: https://images.unsplash.com/photo-1707007939503-601df6e27e58?crop=entropy&cs=srgb&fm=jpg&ixid=M3w2NDU1OTF8MHwxfHJhbmRvbXx8fHx8fHx8fDE3NDMwODM0ODd8&ixlib=rb-4.0.3&q=85&w=1920&h=1080
published:
---

# Processor

**javax.annotation.processing.Processor** : Java提供的注解处理器，在进行代码编译时，虚拟机会去找到对应**Classpath** 路径下面 **META-INF/services** javax-annotation.processing.Processpr 文件中标记的类进行创建并且执行

## 1. 例子

模块A：

```maven
<!--引入谷歌的 autoService注解，目的是可以自动写入到 META/services 文件中-->
<dependency>
    <groupId>com.google.auto.service</groupId>
    <artifactId>auto-service</artifactId>
    <version>1.0-rc6</version>
</dependency>
```

创建一个注解

```java
@Retention(RetentionPolicy.CLASS)
@Target(ElementType.TYPE)
@Documented
public @interface Spi {

}
```

**SupportedAnnotationTypes** ：表示当前Processor 只处理 Spi注解标记的类

```java
@AutoService(Processor.class)
@SupportedAnnotationTypes(AutoSpiProcessor.SPI)
public class AutoSpiProcessor extends AbstractProcessor {
    
    //支持的注解
    public static final String SPI = "com.zhj.starter.annotation.Spi";
    
	@Override
    public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {
        System.out.println("==================================");
        System.out.println("代码编译器执行.................");
        System.out.println("==================================");
        return false;
    }
}

```

创建一个模块B：引入上面的依赖A，创建一个标记注解

![1656491294623](images/1656491294623.png)

![1656491229500](images/1656491229500.png)



## 2. 工具类

### 2.1 ProcessionEnvironment

是一个注解处理工具的集合，常用有以下：

- Types：操作类型，例如：判断一个类型是否是另外一个类型的子类
- Filer：用于创建新的源，类或者辅助文件的管理器
- Elements：返回用于操作元素的工具

### 2.2 Element

**Element** 是一个接口，表示一个程序元素，它可以是包、类、方法或者一个变量。

![1656554023300](images/1656554023300.png)

- PackageElement：表示一个程序包元素，提供对有关包及其成员信息的访问
- ExecutableElement：表示某个类或者接口的方法、构造方法或初始化程序（静态或者实例），包括注释类型的元素
- TypeElement：表示一个类或者接口程序元素，提供对有关类型及其成员的信息的访问。枚举类型是一种类，而注解是一种接口类型
- VariableElement：表示一个字段、enum常量、方法或构造方法参数、局部变量或异常参数

