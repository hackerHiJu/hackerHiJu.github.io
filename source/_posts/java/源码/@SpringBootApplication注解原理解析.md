---
title: "@SpringBootApplication注解原理解析"
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
thumbnail: https://images.unsplash.com/photo-1682686578023-dc680e7a3aeb?crop=entropy&cs=srgb&fm=jpg&ixid=M3w2NDU1OTF8MHwxfHJhbmRvbXx8fHx8fHx8fDE3NDMxNTE3MjJ8&ixlib=rb-4.0.3&q=85&w=1920&h=1080
published: false
---

# @SpringBootApplication

SpringBoot的启动注解，是一个组合注解，其中包括了以下几个注解：

- SpringBootConfiguration：用于标注启动类是一个配置类
  - Configuration：配置类注解
  - Indexed：用于提高类的解析效率，项目编译打包时会自动生成 **META-INF/spring.components** 文件，会被 **CandidateComponentsIndexLoader** 读取并且加载，转换为 **CandidateComponentsIndex** 对象
- EnableAutoConfiguration：springboot用于自动配置的重点注解
  - AutoConfigurationPackage：导入了 **AutoConfigurationPackages.Registrar**
  - Import：导入了 **AutoConfigurationImportSelector**
- ComponentScan：扫描组件
  - TypeExcludeFilter：扫描自定义的数据
  - AutoConfigurationExcludeFilter：扫描出配置类以及自动配置类



## 1. EnableAutoConfiguration

自动配置类注解，其中导入了 **AutoConfigurationImportSelector** 

