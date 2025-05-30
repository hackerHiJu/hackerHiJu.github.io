---
title: 常用工具以及Gin使用（六）
date: 2025-03-28 05:43:12
updated: 2025-03-28 05:43:12
tags:
  - GO
comments: false
categories: GO
thumbnail: https://images.unsplash.com/photo-1526666923127-b2970f64b422?crop=entropy&cs=srgb&fm=jpg&ixid=M3w2NDU1OTF8MHwxfHJhbmRvbXx8fHx8fHx8fDE3NDMxNTQ5OTJ8&ixlib=rb-4.0.3&q=85&w=1920&h=1080
published: true
---
# 1. Http标准库

## 1.1 http客户端

```go
func main() {
	response, err := http.Get("https://www.imooc.com")
	if err != nil {
		return
	}
	defer response.Body.Close()
	bytes, err := httputil.DumpResponse(response, true)
	if err != nil {
		return
	}
	fmt.Printf("%s", bytes)
}
```

## 1.2 自定义请求头

```go
func main() {
	request, err := http.NewRequest(http.MethodGet, "https://www.imooc.com", nil)
	if err != nil {
		return
	}
	//自定义请求头
	request.Header.Add("header", "value")
	response, err := http.DefaultClient.Do(request)
	if err != nil {
		return
	}
	defer response.Body.Close()
	bytes, err := httputil.DumpResponse(response, true)
	if err != nil {
		return
	}
	fmt.Printf("%s", bytes)
}
```

## 1.3 检查请求重定向

```go
//检查重定向函数
	client := http.Client{CheckRedirect: func(req *http.Request, via []*http.Request) error {
		// via: 所有重定向的路径
		// req: 当前重定向的路径
		return nil
	}}
	response, err := client.Do(request)
	if err != nil {
		return
	}
```

## 1.4 http服务器性能分析

图形界面的使用需要安装 graphviz

- 导入 ：_ "net/http/pprof"  , 下划线代表只使用其中的依赖，不加就会编译报错
- 访问：/debug/pprof
- 使用：
  - go tool pprof http://localhost:8888/debug/pprof/profile 可以查看30秒的cpu使用率
  - go tool pprof http://localhost:6060/debug/pprof/block 查看gorountine阻塞配置文件

# 2. JSON数据处理

## 2.1 实体序列化

```go
type Order struct {
	ID string
	Name string
	Quantity int
	TotalPrice float64
}

func main() {
	o := Order{ID: "1234", Name: "learn go", Quantity: 3, TotalPrice: 30.0}
	fmt.Printf("%+v\n", o)
	//序列化后的字节切片，
	bytes, err := json.Marshal(o)
	if err != nil {
		return
	}
	fmt.Printf("%s\n", bytes)
}
```

**注意：首写字母为小写，Marshal不会进行序列化**

## 2.2 处理字段为小写下划线

使用属性标签

```go
type Order struct {
	ID string `json:"id""`
	Name string `json:"name"`
	Quantity int `json:"quantity"`
	TotalPrice float64 `json:"total_price"`
}

func main() {
	o := Order{ID: "1234", Name: "learn go", Quantity: 3, TotalPrice: 30.0}
	fmt.Printf("%+v\n", o)
	//序列化
	bytes, err := json.Marshal(o)
	if err != nil {
		return
	}
	fmt.Printf("%s\n", bytes)
}
```

## 2.3 省略空字段

在字段上添加 **omitempty** 可以省略空字的字段

```go
type Order struct {
	ID string `json:"id""`
	Name string `json:"name,omitempty"`
	Quantity int `json:"quantity"`
	TotalPrice float64 `json:"total_price"`
}
```

## 2.4 反序列化

```go
func main() {
	//反序列化
	str := `{"id":"1234","name":"learn go","quantity":3,"total_price":30}`
	order := unmarshal[Order](str, Order{})
	fmt.Printf("%+v\n", order)
}
//使用泛型的方法，可以解析出对应的实体类
func unmarshal[T any](str string, t T) any {
	err := json.Unmarshal([]byte(str), &t)
	if err != nil {
		return nil
	}
	return t
}
```

# 3. 自然语言处理

可以调用阿里云的自然语言处理api进行数据的处理

## 3.1 使用Map处理

```go
func mapUnmarshall() {
	str := `{
		"data": [
			{
				"id": 0,
				"word": "请",
				"tags": [
					"基本词-中文"
				]
			},
			{
				"id": 1,
				"word": "输入",
				"tags": [
					"基本词-中文",
					"产品类型修饰词"
				]
			},
			{
				"id": 2,
				"word": "文本",
				"tags": [
					"基本词-中文",
					"产品类型修饰词"
				]
			}
		]
	  }`
	//map存储数据都使用interface来存储
	m := make(map[string]any)
	err := json.Unmarshal([]byte(str), &m)
	if err != nil {
		return
	}
	//如果需要取id为2的数据，需要指明所取的值是一个切片 使用type assertion,包括取后续的数据的时候都要指定类型
	fmt.Printf("%+v\n", m["data"].([]any)[2].(map[string]any)["tags"])
}
```

## 3.2 定义实体处理

```go
//map存储数据都使用interface来存储
m := struct {
    Data []struct{
        Id   int32    `json:"id"`
        Word string   `json:"word"`
        Tags []string `json:"tags"`
    } `json:"data"`
}{}
err := json.Unmarshal([]byte(str), &m)
if err != nil {
    return
}
fmt.Printf("%+v\n", m.Data[2].Tags)
```

# 4. http框架

## 4.1 gin

下载依赖：**go get -u github.com/gin-gonic/gin**、**go get -u go.uber.org/zap** (日志库)

### 4.1.1 启动服务

```go
func main() {
	r := gin.Default()
	r.GET("/ping", func(c *gin.Context) {
		c.JSON(200, gin.H{
			"message": "pong",
		})
	})
	r.Run() // listen and serve on 0.0.0.0:8080
}
```

### 4.1.2 middleware

**Context** 结构体其中包含了请求相关的信息

可以为web服务添加 **“拦截器”** ，添加 middleware 拦截请求打印自己需要的日志

```go
logger, _ := zap.NewProduction()
r.Use(printRequestLog, printHello)
//如果添加多个,先定义上方法，直接添加即可
func printRequestLog(c *gin.Context) {
	logger.Info("Incoming request", zap.String("path", c.Request.URL.Path))
	//放行，如果不释放，后续就不能进行处理
	c.Next()
    //获取到response对象
	logger.Info("处理状态：", zap.Int("status", c.Writer.Status()))
}
func printHello(c *gin.Context) {
	fmt.Println("hello:", c.Request.URL.Path)
	//放行，如果不释放，后续就不能进行处理
	c.Next()
}
```

### 4.1.3 设置请求ID

```go
func setRequestId(c *gin.Context) {
	c.Set("requestId", rand.Int())
	c.Next()
}
```



