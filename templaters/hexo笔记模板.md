<%* 
let url = await tp.web.random_picture("1920x1080", "science fiction,scenery");
let result = url.match(/!\[(.*?)\]\((.*?)\)/);
-%>
---
title: 
date: <% tp.date.now("yyyy-MM-DD hh:mm:ss") %>
updated: <% tp.date.now("yyyy-MM-DD hh:mm:ss") %>
tags:
 - 
comments: false
categories:
  - 
thumbnail: <% result[2] %>
published: false
---