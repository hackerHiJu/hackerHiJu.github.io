/**
 * 修复文章相对路径图片
 * 1. after_post_render: hexo-renderer-marked 把 images/xxx.png 渲染成 "/images/xxx.png"
 *    我们重写为 "/posts/<abbrlink>/images/xxx.png"
 * 2. after_generate: 复制图片文件到输出目录
 */
'use strict';

const fs = require('fs');
const path = require('path');

// 重写文章 content 中的图片路径
// marked 渲染后格式: src="/images/xxx.png"
// 改为: src="/posts/<abbrlink>/images/xxx.png"
hexo.extend.filter.register('after_post_render', function (data) {
  if (!data.content) return data;
  const abbrlink = data.abbrlink || data.slug;
  if (!abbrlink) return data;

  // 匹配 "/images/ （注意 marked 加了前导斜杠）
  data.content = data.content.replace(
    /(["'])\/images\//g,
    '$1/posts/' + abbrlink + '/images/'
  );

  return data;
});

// 复制图片到输出目录
hexo.extend.filter.register('after_generate', function () {
  const publicDir = this.public_dir;
  const sourceDir = this.source_dir;

  function walkPosts(dir) {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (entry.name === 'images') continue;
        walkPosts(fullPath);
      } else if (entry.isFile() && entry.name.endsWith('.md')) {
        const content = fs.readFileSync(fullPath, 'utf8');
        const abbrMatch = content.match(/^abbrlink:\s*(\S+)/m);
        if (!abbrMatch) continue;
        const abbrlink = abbrMatch[1];

        const postDir = path.dirname(fullPath);
        const imagesDir = path.join(postDir, 'images');
        if (!fs.existsSync(imagesDir)) continue;

        const destDir = path.join(publicDir, 'posts', abbrlink, 'images');
        if (!fs.existsSync(destDir)) {
          fs.mkdirSync(destDir, { recursive: true });
        }

        const files = fs.readdirSync(imagesDir);
        for (const file of files) {
          const srcFile = path.join(imagesDir, file);
          const destFile = path.join(destDir, file);
          if (fs.statSync(srcFile).isFile()) {
            fs.copyFileSync(srcFile, destFile);
          }
        }
      }
    }
  }
  walkPosts(path.join(sourceDir, '_posts'));
});
