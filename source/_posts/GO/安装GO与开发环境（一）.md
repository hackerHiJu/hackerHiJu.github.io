---
title: 安装GO与开发环境（一）
date: 2025-03-28 05:13:08
updated: 2025-03-28 05:13:08
tags:
  - GO
comments: false
categories: GO
thumbnail: https://images.unsplash.com/photo-1682687218982-6c508299e107?crop=entropy&cs=srgb&fm=jpg&ixid=M3w2NDU1OTF8MHwxfHJhbmRvbXx8fHx8fHx8fDE3NDMxNTMxODh8&ixlib=rb-4.0.3&q=85&w=1920&h=1080
published: true
---
# 1. GO环境安装

## 1.1 环境搭建

 https://golang.google.cn/dl/  下载win10安装包，(https://studygolang.com/dl)

![1652425097117](https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/1652425097117.png)

![1652425142132](https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/1652425142132.png)

![1652425149882](https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/1652425149882.png)

选择 D盘或者C盘进行安装

![1652425178040](https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/1652425178040.png)

![1652425189957](https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/1652425189957.png)

## 1.2 查看安装

![1652498611329](https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/1652498611329.png)

输出以上的信息就是安装成功了

## 1.3 配置代理

>go env -w GO111MODULE=on
>go env -w GOPROXY=https://goproxy.cn,direct

输入以下信息就可以了

![1652498703708](https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/1652498703708.png)

![1652498756776](https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/1652498756776.png)

# 2. Goland安装

## 2.1 软件安装

https://www.jetbrains.com.cn/go/download/other.html  选择 2021.2.2版本下载，便于后续安装无限制重置试用时间

![1652425359917](https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/1652425359917.png)

![1652425328192](https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/1652425328192.png)

直接点击试用

![1652425512267](https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/1652425512267.png)



## 2.2 插件安装

https://plugins.zhile.io  安装 IDE Eval Reset 插件，后续就可以无限制重置试用期了

![1652425532418](https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/1652425532418.png)

![1652425612697](https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/1652425612697.png)

## 2.3 项目创建以及配置

### 2.3.1 创建GOPATH

在任意盘新建一个文件夹，用于存放编译的代码以及拉取的包数据 **（GOMOD默认存储依赖路径）**

![1652428298762](https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/1652428298762.png)

添加环境变量

![1652428402429](https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/1652428402429.png)

### 2.3.2 创建项目

设置代理，因为网络防火墙的存在，可能导致 go 在拉取第三方包时无法直接通过 go get 拉取，通过 GOPROXY 的中间代理来拉取包

![1652428575186](https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/1652428575186.png)



![1652426006926](https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/1652426006926.png)

勾选是 index entire GOPATH以所有整个GOPATH，不然无法导入包

![1652428658380](https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/1652428658380.png)

![1652428757946](https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/1652428757946.png)

![1652426120769](https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/1652426120769.png)

### 2.3.3 问题说明

注意创建 go 文件时自动生成的包名，需要设置为 main 包进行执行

![1652429402154](https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/1652429402154.png)

## 2.4 路径说明

- GOROOT：为golang的安装路径，安装好golang之后就有了
- GOPATH：
  - 存放sdk以外的第三方库
  - 自己收藏的可复用的代码
  - 其中$GOPATH目录约定三个子目录：
    - src存放源码
    - pkg编译时生成的中间文件
    - bin编译后生成的可执行文件

Goland中Project GOPATH以及Global GOPATH

- Project GOPATH：只有当前这一个项目可以使用
- Global GOPATH：所有的项目都可以使用，也可以在环境变量中配置

# 3. 安装Idea+GO插件

安装GO的插件，后续就跟以上操作一样了

![1652498981079](https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/1652498981079.png)

![1652499009188](https://cdn.jsdelivr.net/gh/hackerHiJu/note-picture@main/note-picture/1652499009188.png)

