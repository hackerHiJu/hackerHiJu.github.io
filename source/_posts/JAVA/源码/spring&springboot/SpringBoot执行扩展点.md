---
title: SpringBoot执行扩展点
date: 2025-04-18 01:12:20
updated: 2025-04-18 01:12:20
tags: 
comments: false
categories: 
thumbnail: https://images.unsplash.com/photo-1500964757637-c85e8a162699?crop=entropy&cs=srgb&fm=jpg&ixid=M3w2NDU1OTF8MHwxfHJhbmRvbXx8fHx8fHx8fDE3NDQ5NTMxNDB8&ixlib=rb-4.0.3&q=85&w=1920&h=1080
published: false
---
## 1. BootstrapRegistryInitializer

- initialize：当创建 BootstrapContext 上下文对象时进行初始化的调用，可以注册一些特殊的实例获取器，以及添加BootstrapContextClosedEvent关闭的事件监听器

## 2. SpringApplicationRunListeners

- starting：容器正在启动时的监听器
- environmentPrepared