---
title: NodeRed基础入门到实践
date: 2024-12-30 19:07:05
updated: 2024-12-30 19:07:05
tags:
  - 前端
  - 物联网
comments: true
categories:
  - 前端
  - 物联网
thumbnail: https://images.unsplash.com/photo-1505322022379-7c3353ee6291?crop=entropy&cs=srgb&fm=jpg&ixid=M3w2NDU1OTF8MHwxfHJhbmRvbXx8fHx8fHx8fDE3NDMwNzYzOTN8&ixlib=rb-4.0.3&q=85&w=1920&h=1080
---

# Node-Red使用笔记

Node-RED是IBM推出的一款基于流程编程的视觉化程式设计语言开发工具，通过这一工具，开发者可以将硬件设备与应用程序接口和在线服务相连，并组成一个小型物联网。

# 1. 服务安装

> docker run -d --name node-red -p 1880:1880 nodered/node-red:latest

进入容器更换Node的源

> npm config set registry https://registry.npmmirror.com/

启动完成访问 **127.0.0.1:1880**



# 2. 节点使用

整个 **Node-Red中都是以msg作为顶层的数据体**，如果后续接受的数据格式不一样，则可以使用 **json数据转换节点** 将其转换为正确的节点

```json
{
  "msg": {}
}
```

### 2.1 通用节点

#### 1. inject

Inject标签可以配置一个动态的数据触发入口，当直接点击节点时可以直接触发节点数据

- 勾选上立刻执行，可以指定延迟指定时间后再触发数据
- 点击重复可以再次触发
- 后面的选择框功能：
  - 周期执行：指定每隔一段时间可以执行一次
  - 指定时间段周期性执行：在指定周一到周天在几点到几点之间每隔多少分钟执行
  - 指定时间：指定时间点执行

![image-20241220180628419|1570x1046](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20241220180628419.png)

#### 2. debug

debug节点用于输出数据到控制台或者日志中

- msg：读取json结构体中msg.xxx对象打印到debug窗口
- 与表达式输出相同：会输出整个传入的状态信息，其中包括正常的数据、异常的信息等
- JSONata表达式：输入表达式进行处理

![image-20241220181302178](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20241220181302178.png)

填写表达式

![image-20241224112928351](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20241224112928351.png)



当创建一个 **inject** 后进行触发，并且打印对应的数据日志到控制台

![image-20241220181334463](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20241220181334463.png)

![image-20241220184510490](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20241220184510490.png)

#### 3. complete

complete节点用于监听对应节点的完成，当一个节点完成后会触发执行

![image-20241224111057435](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20241224111057435.png)

通过节点进行关联需要监听哪一个，监听多个时每一个都会执行一次

![image-20241224111126171](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20241224111126171.png)

#### 4. catch

捕获节点执行异常时进行执行对应的流程

![image-20241224111639305](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20241224111639305.png)

- 所有节点：当前流程页中所有节点执行失败了都会进行触发
- 在同一个分组：在当前分组中的节点抛出异常进行触发
- 指定节点：选中多个节点进行触发

![image-20241224112246259](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20241224112246259.png)

#### 5. status

用于监听某个节点的状态值变更信息，例如：tcp的连接、mqtt客户端的连接等，都会输出对应的状态信息

![image-20241224175133998](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20241224175133998.png)

#### 6. link in/link out/link call

连接标签用于连接另外一个流程和接收一个流程的连接，流程1创建 **link out** 表示流程需要连接到另外一个流程中，**link in**表示可以作为一个可链接的流程，**link call** 表示调用其他接点后需要获取到返回值配合 **link out中的返回调用链接节点使用**

- 发送到所有链接的流程：调用其他的流程数据
- 返回调用链接节点：返回调用流程的值



![image-20241224175538380](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20241224175538380.png)

![image-20241224175557598](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20241224175557598.png)

调用子流程并且获取到对应的返回值，例如下面的流程，我输入8888数据，通过**link call** 去调用上面的子流程，子流程调用了一个function2并且将8888修改为999，这时候流程就会打印999数据，如果子流程不将 **link out** 修改为返回调用链接节点，那么就接不到返回值

![image-20241225161449372](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20241225161449372.png)

![image-20241225161433700](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20241225161433700.png)

#### 5. comment

给流程添加注释



### 2.2 功能节点

#### 1. function

函数节点，可以自定义一个js脚本

- 初始化函数：当流程在进行部署时进行执行
- 运行函数：流程正常执行时运行
- 终止函数：将在停止或重新部署节点时运行

#### 2. switch

switch case，匹配对应数据执行对应节点

- 全选所有规则：所有规则都会进行匹配，匹配到了就执行
- 接收匹配到第一条信息后停止：只会匹配成功一次

![image-20241225163640846](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20241225163640846.png)

#### 3. change

查找数据并且进行转换

- 设定值：给某个字段设置值
- 替换值：获取到某个字段并且替换具体的值
- 删除：删除指定字段
- 转移：将指定的字段，设置到另外一个字段中

将type字段中为数据2的数据替换为123456

![image-20241225164417913](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20241225164417913.png)

#### 4. range

将对应的值映射到指定的范围，创建随机函数生成 0 -1

```js
msg.payload = Math.random()
return msg;
```

![image-20241225165051238](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20241225165051238.png)

![image-20241225165138666](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20241225165138666.png)

#### 5. template

创建模板数据，根据 **mustache** 语法来指定转换为对应的格式

![image-20241225165434076](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20241225165434076.png)

![image-20241225165539007](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20241225165539007.png)

#### 6. delay

延迟发送数据到节点

- 延迟每一条信息
  - 固定时间
  - 随机延迟
  - 运行msg.delay复写延迟时间：如果msg.delay指定了时间就依照msg.delay指定，否则就按照默认的
- 限制消息的速率
  - 指定时间发送指定条数的数据
  - 消息存储：
    - 将中间消息排队：后续来的数据存入队列
    - 不传输中间消息：只发送第一次的消息
    - 在第2个输出端发送中间消息：第一个消息发送第一个连接端口，后续的消息都发送到第二个

![image-20241225170605389](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20241225170605389.png)

#### 7. trigger

触发节点，比较核心的配置

- 发送的内容

- 等待被重置
  - 消息中含有msg.reset重置进行发送
  - msg.payload等于某个值时进行触发
- 等待指定时间进行发送
- 周期性重发



### 2.3 网络节点

#### 1. mqtt in

创建mqtt客户端，去监听对应的topic信息

![image-20250117104803713](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20250117104803713.png)

#### 2. mqtt out

输出数据到对应的topic中

![image-20250117104839224](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20250117104839224.png)

#### 3. http in

创建一个http的请求接口，请求的路径是 服务部署的路径+端口/hello，例如：127.0.0.1:1880/echo

![image-20250117105414263](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20250117105414263.png)

#### 4. http response

创建一个响应返回给 http in作为响应对象

![image-20250117105614886](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20250117105614886.png)

![image-20250117111239544](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20250117111239544.png)

需要注意的是，响应的数据必须要返回系统的 **msg** 对象才有效

#### 5. http request

发起自定义的http请求，直接请求上面定义的接口数据

![image-20250117111438516](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20250117111438516.png)

![image-20250117111425808](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20250117111425808.png)

#### 6. websocket in

创建一个websocket的服务端

#### 7. websocket out

创建一个websocket的客户端

### 2.4 序列化

#### 1. split分割

将对应的数据进行分割

- 字符串/buffer：使用自定义的字符、缓冲区或固定长度将消息进行拆分
- 流模式：该节点可以重排消息流
- 数组：消息被拆分为单个数组元素或固定长度的数组

#### 2. join合并

将数据通过对应的key值进行合并，节点会根据返回的topic字段进行数据的合并

- 发送信息：当节点合并到指定数量的数据后发送给下一个节点
- 和每个后续消息：节点会合并上所有的数据的数据进行发送，每收到一次就发送一次

![image-20250117141311846](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20250117141311846.png)

![image-20250117141251735](https://cdn.jsdelivr.net/gh/hackerhaiJu/note-picture@main/note-picture/image-20250117141251735.png)

#### 3. json转换

用于将指定的格式转换为node-red中的json格式，会将格式包装为 **msg.payload** 的形式

- JSON字符串与对象互转
- 总是转换为JSON字符串
- 总是转为JS对象
