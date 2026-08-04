# Hexo 博客全量迁移至安知鱼 (AnZhiYu) 主题方案

## 用户偏好确认
- **主色调**：保留橙红 `#ff6b35`（暗色模式 `#ffd700`）
- **评论系统**：从 Gitalk 切换到 Giscus
- **高级特性**：AI 文章摘要 + 首页双栏布局 + 评论弹幕 + 左下角音乐播放器

---

## 一、主题安装（3步）

### 1. 克隆安知鱼主题
```bash
git clone -b main https://github.com/anzhiyu-c/hexo-theme-anzhiyu.git themes/anzhiyu
rm -rf themes/anzhiyu/.git
```
> 将主题文件直接纳入博客仓库，便于版本管理和 GitHub Actions 构建。

### 2. 安装 Pug 渲染器（安知鱼用 Pug 模板引擎）
```bash
npm install hexo-renderer-pug --save
```
> `hexo-renderer-stylus` 已安装，无需重复安装。

### 3. 安装 hexo-wordcount（字数统计）
```bash
npm install hexo-wordcount --save
```
> 当前项目引用了 word_count 但缺少此依赖，补全后安知鱼字数统计功能才能正常工作。

---

## 二、修改 `_config.yml`（主配置）

仅改动 1 处：
```yaml
theme: anzhiyu   # 原: redefine
```
其余保持不变（permalink `posts/:abbrlink/`、category_map、skip_render、abbrlink 等全部保留）。

---

## 三、创建 `_config.anzhiyu.yml`（主题配置，核心工作量）

从 `themes/anzhiyu/_config.yml` 复制基础模板，然后按以下模块深度定制：

### 3.1 导航菜单 `menu`（安知鱼用嵌套格式）
```yaml
menu:
  首页:
    首页: / || anzhiyu-icon-house
  文章:
    隧道: /archives/ || anzhiyu-icon-box-archive
    分类: /categories/ || anzhiyu-icon-shapes
    标签: /tags/ || anzhiyu-icon-tags
  Java:
    Spring源码: /categories/Spring/ || anzhiyu-icon-fanqie
    Redis: /categories/Redis/ || anzhiyu-icon-db
    Netty: /categories/Netty/ || anzhiyu-icon-network
    K8s: /categories/K8s/ || anzhiyu-icon-ship
    JVM: /categories/JVM/ || anzhiyu-icon-cup
    Mysql: /categories/Mysql/ || anzhiyu-icon-db
    多线程: /categories/concurrent/ || anzhiyu-icon-thread
    Mybatis: /categories/Mybatis/ || anzhiyu-icon-book
    Seata: /categories/Seata/ || anzhiyu-icon-distribute
  AI:
    AI大模型: /categories/AI/ || anzhiyu-icon-ai
    Pytorch: /categories/AI/ || anzhiyu-icon-brain
    显卡性能: /categories/AI/ || anzhiyu-icon-gpu
  GO:
    GO语言: /categories/GO/ || anzhiyu-icon-code
  更多:
    汇编语言: /categories/assembly/ || anzhiyu-icon-cpu
    前端: /categories/frontend/ || anzhiyu-icon-html
    分布式: /categories/distributed/ || anzhiyu-icon-cloud
    协议: /categories/protocol/ || anzhiyu-icon-agreement
    书签: /bookmarks/ || anzhiyu-icon-bookmark
  关于:
    关于我: /about/ || anzhiyu-icon-paper-plane
    GitHub: https://github.com/hyqf98 || anzhiyu-icon-github
    CSDN: https://blog.csdn.net/weixin_43915643 || anzhiyu-icon-link
```

### 3.2 站点信息 & 头像
```yaml
avatar:
  img: /images/头像.jpg
  effect: true
favicon: /images/头像.jpg
```

### 3.3 主题色（保留橙红）
```yaml
theme_color:
  enable: true
  main: "#ff6b35"
  dark_main: "#ffd700"
  paginator: "#ff6b35"
  text_selection: "#ff6b3540"
  link_color: "var(--anzhiyu-fontcolor)"
  meta_color: "var(--anzhiyu-fontcolor)"
  hr_color: "#ff6b3523"
  code_foreground: "#fff"
  code_background: "var(--anzhiyu-code-stress)"
  toc_color: "#ff6b35"
  scrollbar_color: "var(--anzhiyu-scrollbar)"
  meta_theme_color_light: "#fff9f5"
  meta_theme_color_dark: "#18171d"
```

### 3.4 社交链接（GitHub + CSDN）
```yaml
social:
  GitHub: https://github.com/hyqf98 || anzhiyu-icon-github
  CSDN: https://blog.csdn.net/weixin_43915643 || anzhiyu-icon-link
```

### 3.5 首页顶部 `home_top`（安知鱼签名特性）
```yaml
home_top:
  enable: true
  timemode: date
  title: 何忆清风
  subTitle: 全栈开发者 · Java · AI · Go
  siteText: hyqf98.github.io
  category:
    - name: Java
      path: /categories/Java/
      shadow: var(--anzhiyu-shadow-red)
      class: red
      icon: anzhiyu-icon-fanqie
    - name: AI
      path: /categories/AI/
      shadow: var(--anzhiyu-shadow-blue)
      class: blue
      icon: anzhiyu-icon-ai
    - name: GO
      path: /categories/GO/
      shadow: var(--anzhiyu-shadow-green)
      class: green
      icon: anzhiyu-icon-code
  default_descr: 再怎么看我也不知道怎么描述它的啦！
  banner:
    tips: 开源项目
    title: Easy Agent Pilot
    image: https://opengraph.githubassets.com/1/hyqf98/easy_agent_pilot
    link: https://github.com/hyqf98/easy_agent_pilot
```

### 3.6 首页双栏 + 副标题打字机
```yaml
article_double_row: true   # 首页双栏瀑布流
subtitle:
  enable: true
  effect: true
  loop: true
  source: 1             # 调用一言 API
  sub:
    - 要么忙着活，要么忙着死。
    - 废话少说，上号！
    - 代码是写给人看的，只是恰好能在机器上运行。
```

### 3.7 首页 Banner 图
```yaml
index_img: /images/banner-bg.png   # 复用现有壁纸
```

### 3.8 评论系统 — Giscus
```yaml
comments:
  use: Giscus
  text: true
  lazyload: false
  count: true
  card_post_count: false

giscus:
  repo: hyqf98/hyqf98.github.io
  repo_id: # 需用户在 giscus.app 获取后填入
  category: Announcements
  category_id: # 需用户在 giscus.app 获取后填入
  theme:
    light: light
    dark: dark
  option:
    data-lang: zh-CN
    data-mapping: pathname
    data-input-position: top
    data-reactions-enabled: 1
```
> **⚠️ 前置条件**：你需要在 GitHub 仓库启用 Discussions，然后在 https://giscus.app 获取 `repo_id` 和 `category_id` 填入。我会用 `gh` CLI 帮你启用 Discussions。

### 3.9 评论弹幕
```yaml
comment_barrage_config:
  enable: true
  maxBarrage: 1
  barrageTime: 4000
  accessToken: ""  # 配合 Giscus 时可留空
  mailMd5: ""
```

### 3.10 AI 文章摘要
```yaml
post_head_ai_description:
  enable: true
  gptName: 何忆清风
  mode: local        # 本地模式免费，基于文章内容提取摘要
  switchBtn: false
  randomNum: 3
  basicWordCount: 1000
```

### 3.11 左下角音乐播放器
```yaml
nav_music:
  enable: true
  console_widescreen_music: false
  id: 8152976493     # 默认网易云歌单（后续可改为你自己的歌单ID）
  server: netease
  volume: 0.7
  all_playlist: https://y.qq.com/n/ryqq/playlist/8802438608
```

### 3.12 搜索（保留本地搜索）
```yaml
local_search:
  enable: true
  preload: true
```

### 3.13 Mermaid 图表
```yaml
mermaid:
  enable: true
  theme:
    light: default
    dark: dark
```

### 3.14 侧边栏卡片（作者信息）
```yaml
aside:
  enable: true
  position: right
  mobile: true
  card_author:
    enable: true
    description: >
      <div>全栈开发者，8年码龄。专注于 <b>Java/Spring 生态、AI/深度学习、Go、Rust</b>。</div>
      <div>热爱源码分析，分享技术心得。CSDN 博客 <b>440万+</b> 访问量。</div>
    name_link: /
  card_announcement:
    enable: true
    content: 欢迎来到我的技术小窝~
  card_recent_post:
    enable: true
    limit: 5
  card_tags:
    enable: true
    limit: 40
  card_archives:
    enable: true
    type: monthly
    limit: 8
  card_webinfo:
    enable: true
    post_count: true
```

### 3.15 字体配置
```yaml
font:
  global-font-size: 16px
  code-font-family: consolas, Menlo, "PingFang SC", "Microsoft JhengHei", "Microsoft YaHei", sans-serif

blog_title_font:
  font-family: PingFang SC, 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif
```
> 保持与当前相同的字体栈，不做激进改动。

### 3.16 预加载动画
```yaml
preloader:
  enable: true
  source: 3       # 使用安知鱼内置加载动画
  avatar: /images/头像.jpg
```

### 3.17 高级特效（默认开启的安知鱼特色）
```yaml
pjax:
  enable: true
lazyload:
  enable: true
  field: site
  blur: true
fancybox: true
instantpage: true
snackbar:
  enable: true
  position: bottom-right
  bg_light: "#ff6b35"
  bg_dark: "#18171d"
darkmode:
  enable: true
  button: true
  autoChangeMode: 1
universe:
  enable: true     # 深色模式粒子效果
translate:
  enable: true     # 简繁转换
readmode: true
centerConsole:
  enable: true
```

### 3.18 页脚
```yaml
footer:
  owner:
    enable: true
    since: 2025
  custom_text: HiJu · 何忆清风
  runtime:
    enable: true
    launch_time: 03/24/2025 18:00:00
  socialBar:
    enable: true
    left:
      - title: GitHub
        link: https://github.com/hyqf98
        icon: anzhiyu-icon-github
      - title: CSDN
        link: https://blog.csdn.net/weixin_43915643
        icon: anzhiyu-icon-link
  footerBar:
    enable: true
    authorLink: /
    cc:
      enable: true
      link: /copyright
    linkList:
      - link: https://github.com/anzhiyu-c/hexo-theme-anzhiyu
        text: 主题
      - link: https://github.com/hyqf98
        text: GitHub
```

### 3.19 文章版权
```yaml
post_copyright:
  enable: true
  decode: false
  location: 四川
  license: CC BY-NC-SA 4.0
  license_url: https://creativecommons.org/licenses/by-nc-sa/4.0/
```

### 3.20 封面图设置（复用现有 14 张分类封面）
```yaml
cover:
  index_enable: true
  aside_enable: true
  archives_enable: true
  position: both
  default_cover:
    - /images/covers/Java.jpg
    - /images/covers/AI.jpg
    - /images/covers/GO.jpg
    - /images/covers/Netty.jpg
    - /images/covers/Spring.jpg
    - /images/covers/Redis.jpg
```
> 现有文章 front-matter 中的 `cover` 字段优先级更高，会覆盖默认值。

### 3.21 Inject 自定义（保留导航橙色文字适配）
```yaml
inject:
  head:
    - <style>.anzhiyu-navbar a { color: inherit !important; }</style>
```

---

## 四、新建页面

### 4.1 关于页 `/about/`
```bash
hexo new page about
```
编辑 `source/about/index.md`，包含：
- 个人简介（全栈开发者，Java/AI/Go，8年码龄）
- 技术栈列表
- 开源项目展示（easy_agent_pilot、dataset_manager、easy_db_mcp_server、easy_terminal）
- CSDN 博客成就（440万+访问，134篇原创，2894粉丝）
- 联系方式（GitHub + CSDN 链接）

### 4.2 书签页 `/bookmarks/`（迁移现有数据）
用安知鱼 `{% flink %}` 标签外挂替代，将 `source/_data/bookmarks.yml` 的 10 个分类数据转换为 flink 格式：
```markdown
---
title: 这里有好东西
top_img: /images/banner-bg.png
---

{% flink %}
- class_name: AI 工具
  class_desc: AI 对话与创作工具
  flink_style: anzhiyu
  link_list:
    - name: ChatGPT
      link: https://chat.openai.com
      descr: OpenAI 的 AI 对话助手
      avatar: https://chat.openai.com/favicon.ico
    # ... 全部迁移
{% endflink %}
```

### 4.3 版权页 `/copyright/`（安知鱼 footerBar 引用）
创建简单的 CC BY-NC-SA 4.0 版权声明页。

### 4.4 保留现有页面
- `source/categories/index.md` → 补充 `type: categories`（已存在）
- `source/tags/index.md` → 补充 `type: tags`（已存在）
- `source/archives/` → 安知鱼自动生成，无需手动创建

---

## 五、文章 front-matter 兼容性

**✅ 无需改动**：安知鱼使用与 redefine 相同的标准 Hexo front-matter：
- `title` → 兼容
- `date` / `updated` → 兼容
- `categories`（块列表格式）→ 兼容
- `tags` → 兼容
- `cover` → 兼容
- `abbrlink` → 兼容（URL 不变）

**⚠️ 可选新增字段**（不强制，有则更美观）：
- `main_color: '#3e5658'` — 手动指定文章主色调（未指定时 AI 摘要/顶部色条使用全局主题色）
- `description` — 文章描述（用于首页摘要）

---

## 六、Giscus 评论系统前置操作

迁移代码前/后我会帮你执行：
```bash
gh api repos/hyqf98/hyqf98.github.io -X PATCH -f has_discussions=true
```
然后你需要去 **https://giscus.app** 完成最后2步：
1. 输入仓库 `hyqf98/hyqf98.github.io`
2. 复制生成的 `data-repo-id` 和 `data-category-id` 填入 `_config.anzhiyu.yml`

---

## 七、清理工作

### 7.1 保留 redefine 作为回退
- 不卸载 `hexo-theme-redefine` npm 包
- 保留 `_config.redefine.yml`（万一想切回去）
- 仅改 `_config.yml` 中的 `theme:` 值

### 7.2 书签数据迁移后
- 旧的 `source/_data/bookmarks.yml` 保留作为数据备份
- bookmarks 页面内容改为 flink 格式

---

## 八、执行顺序

| 步骤 | 操作 | 风险 |
|---|---|---|
| 1 | 克隆安知鱼主题到 `themes/anzhiyu/` | 无 |
| 2 | `npm install hexo-renderer-pug hexo-wordcount` | 无 |
| 3 | 创建 `_config.anzhiyu.yml`（深度定制配置） | 低 |
| 4 | 改 `_config.yml` 的 `theme: anzhiyu` | 可逆 |
| 5 | 新建 about / copyright 页面 | 无 |
| 6 | 转换 bookmarks 页面为 flink 格式 | 数据备份安全 |
| 7 | `gh api` 启用 Discussions | 无 |
| 8 | `hexo clean && hexo g && hexo s` 本地验证 | 构建报错则排查 |
| 9 | 等用户填入 Giscus ID 后最终测试 |  |
| 10 | 等用户确认后 git 提交推送 |  |

---

## 九、风险与对策

| 风险 | 对策 |
|---|---|
| 安知鱼主题与 Hexo 7.3.0 兼容性 | 安知鱼要求 Hexo ≥5.3.0，7.3.0 兼容 |
| 分类路径与 redefine 不同 | 安知鱼的 category 页面是层级结构，导航菜单直接指向 `/categories/Spring/` 等一级分类页 |
| Giscus 需要用户手动获取 ID | 已标注为前置条件，不影响其余迁移 |
| 音乐播放器默认歌单不是你的 | 后续可改为你的网易云歌单 ID |
| 安知鱼内置图标库可能与部分 Font Awesome 不同 | 导航图标改用 anzhiyu-icon 系列 |

---

## 十、产出物清单

1. `themes/anzhiyu/` — 安知鱼主题完整代码
2. `_config.anzhiyu.yml` — 深度定制的主题配置（橙红配色、导航菜单、AI 摘要、双栏、音乐等）
3. `_config.yml` — `theme: anzhiyu`（仅改此一行）
4. `package.json` — 新增 `hexo-renderer-pug`、`hexo-wordcount`
5. `source/about/index.md` — 关于页面（含个人简介+开源项目）
6. `source/bookmarks/index.md` — 书签页改为 flink 格式
7. `source/copyright/index.md` — 版权声明页
8. GitHub Discussions 已启用（为 Giscus 准备）
9. `hexo g` 构建通过，本地 localhost:4000 验证效果
