---
title: Gorm框架基本使用（四）
date: 2025-03-28 05:41:38
updated: 2025-03-28 05:41:38
tags:
  - GO
comments: false
categories: GO
thumbnail: https://images.unsplash.com/photo-1682687220499-d9c06b872eee?crop=entropy&cs=srgb&fm=jpg&ixid=M3w2NDU1OTF8MHwxfHJhbmRvbXx8fHx8fHx8fDE3NDMxNTQ4OTh8&ixlib=rb-4.0.3&q=85&w=1920&h=1080
published: true
---
# 1. 下载 Gorm 依赖

## 1.1 gorm框架

> go get -u gorm.io/gorm 

## 1.2 mysql驱动依赖

> go get -u gorm.io/driver/mysql

注意如果下载不了依赖：**go env -w GOPROXY=https://goproxy.cn,direct** 修改依赖代理网站

# 2. 快速开始

```go
type Author struct {
	Email string
}
//指定当前实例的表名称，默认使用实例结构的 蛇型作为表名
func (u Userinfo) TableName() string {
	return "user_info"
}

type Userinfo struct {
	gorm.Model  //使用 gorm.Model提供的实体，这里会将字段自动展开，封装了通用字段，例如id
	Author `gorm:"embeded"` //可以对正常的内嵌对象通过 embeded 标签嵌入  embeddedPrefix:author_ 增加前缀
	Name string
	Gender string
	Hobby string
}

func main() {
    //创建连接获取 db对象
	dsn := "root:root@tcp(localhost:3306)/db1?charset=utf8mb4&parseTime=True&loc=Local"
	db, err := gorm.Open(mysql.Open(dsn), &gorm.Config{})
	if err != nil {
		panic(err)
	}
	u1 := Userinfo{Name: "张四", Gender: "男", Hobby: "学习"}
    //如果没有表，会自动创建，如果存在表，就会执行插入数据
	db.Create(&u1)
    //打印返回的ID
	fmt.Println(u1.ID)

}
```

# 3. 创建

## 3.1 实体

```go
type Author struct {
	Email string
}

type Userinfo struct {
	gorm.Model  //使用 gorm.Model提供的实体，这里会将字段自动展开
	Author `gorm:"embeded"` //可以对正常的内嵌对象通过 embeded 标签嵌入  embeddedPrefix:author_ 增加前缀
	Name string
	Gender string
	Hobby string
}
```



## 3.1 插入指定字段

```go
func CreateAndUpdate(db *gorm.DB) {
	u1 := Userinfo{Name: "张四", Gender: "男", Hobby: "学习", Author: Author{Email: "1424132555@qq.com"}}
	//根据给出的字段插入数据
	result := db.Select("Name", "Email", "Hobby").Create(&u1)
	fmt.Printf("ID:%d, Error:%v, 插入记录行数:%d \n", u1.ID, result.Error, result.RowsAffected)
	// INSERT INTO `user_info` (`name`,`gender`,`email`) VALUES ("jinzhu", 18, "1424132555@qq.com")
}
```

## 3.2 忽略字段

```go
func CreateAndOmit(db *gorm.DB) {
	u1 := Userinfo{Name: "张四", Gender: "男", Hobby: "学习", Author: Author{Email: "1424132555@qq.com"}}
	//忽略指定的字段
	result := db.Omit("Name", "Hobby").Create(&u1)
	fmt.Printf("ID:%d, Error:%v, 插入记录行数:%d \n", u1.ID, result.Error, result.RowsAffected)
}
```

## 3.3 批量插入

```go
//也可以指定每个会话的批次大小：db := db.Session(&gorm.Session{CreateBatchSize: 1000})
//指定所有全局的大小：db, err := gorm.Open(mysql.Open(dsn), &gorm.Config{ CreateBatchSize:1000,})

func batchInsert(db *gorm.DB)  {
	users := []Userinfo{
		{Name: "李四"},
		{Name: "张三"},
	}
	//指定每批的大小
	db.CreateInBatches(users, 100)
    
	db.Create(users)
	for _, i := range users {
		fmt.Println("id:", i.ID)
	}
}
```

## 3.4 创建钩子

GORM允许用户自定义钩子：

- BeforeSave：在保存之间
- BeforeCreate：创建之前
- AfterSave：保存之后
- AfterCreate：创建之后

