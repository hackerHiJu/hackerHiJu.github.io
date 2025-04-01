<%* 
let url = await tp.web.random_picture("3840x2160", "science fiction,scenery");
let result = url.match(/!\[(.*?)\]\((.*?)\)/);
-%>
<% result[2] %>