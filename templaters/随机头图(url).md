<%* 
let url = await tp.web.random_picture("1920x1080", "science fiction,scenery");
let result = url.match(/!\[(.*?)\]\((.*?)\)/);
-%>
<% result[2] %>