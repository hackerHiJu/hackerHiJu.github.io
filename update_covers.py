# -*- coding: utf-8 -*-
import os, re
from pathlib import Path

posts_dir = Path("source/_posts")

# 目录路径前缀 -> 封面映射（按精确度从长到短匹配）
DIR_RULES = [
    ("AI/StableDiffusion",       "AI.jpg"),
    ("AI/大模型",                 "AI.jpg"),
    ("技术/GO",                   "GO.jpg"),
    ("技术/JVM",                  "JVM.jpg"),
    ("技术/后端/SpringCloud",     "SpringCloud.jpg"),
    ("技术/后端/Spring",          "Spring.jpg"),
    ("技术/后端/Netty",           "Netty.jpg"),
    ("技术/后端/Mybatis",         "Mybatis.jpg"),
    ("技术/后端/Java基础",        "Java.jpg"),  # 下面会按内容细分并发类
    ("技术/中间件/Mysql",         "Mysql.jpg"),
    ("技术/中间件/Redis",         "Redis.jpg"),
    ("技术/中间件/RabbitMq",      "RabbitMq.jpg"),
    ("技术/中间件/RocketMQ",      "RocketMQ.jpg"),
    ("技术/中间件/Seata",         "Seata.jpg"),
    ("技术/中间件/ShardingJdbc",  "ShardingJdbc.jpg"),
    ("技术/中间件/TDengine",      "TDengine.jpg"),
    ("技术/运维/K8s",             "K8s.jpg"),
    ("技术/前端",                 "Frontend.jpg"),
    ("技术/协议",                 "Http.jpg"),
    ("技术/汇编",                 "汇编语言.jpg"),
    ("摄影",                      "摄影.jpg"),
    ("更多",                      "Tools.jpg"),
]

# Java基础下并发主题文章用 并发.jpg
CONCURRENT_KEYWORDS = ["并发", "CompletableFuture", "AQS", "线程池"]
CONCURRENT_FILES = ["并发编程", "线程池", "AQS源码", "CompletableFuture源码"]

def get_cover(rel_path, title):
    """根据相对路径和标题确定封面"""
    # 精确匹配目录规则
    for prefix, cover in DIR_RULES:
        if rel_path.startswith(prefix + "/") or rel_path == prefix:
            # Java基础并发细分
            if prefix == "技术/后端/Java基础":
                fname = Path(rel_path).stem
                if any(k in fname for k in CONCURRENT_FILES):
                    return "并发.jpg"
            return cover
    return None

def update_frontmatter(filepath, cover_file):
    """更新 frontmatter 中的 cover 和 thumbnail"""
    text = filepath.read_text(encoding="utf-8")
    cover_path = f"/images/covers/{cover_file}"
    changed = False
    
    # 更新 cover:
    new_text = re.sub(
        r'^(cover:\s*).+$',
        lambda m: m.group(1) + cover_path,
        text, count=1, flags=re.MULTILINE
    )
    if new_text != text:
        text = new_text; changed = True
    
    # 更新 thumbnail:
    new_text = re.sub(
        r'^(thumbnail:\s*).+$',
        lambda m: m.group(1) + cover_path,
        text, count=1, flags=re.MULTILINE
    )
    if new_text != text:
        text = new_text; changed = True
    
    # 如果没有 thumbnail 字段，在 cover 后面加上
    if not re.search(r'^thumbnail:', text, re.MULTILINE):
        text = re.sub(
            r'(^cover:\s*.+\n)',
            r'\1' + f'thumbnail: {cover_path}\n',
            text, count=1, flags=re.MULTILINE
        )
        changed = True
    
    if changed:
        filepath.write_text(text, encoding="utf-8")
    return changed

# 遍历处理
results = {"updated": [], "skipped": [], "no_cover": []}
for md in sorted(posts_dir.rglob("*.md")):
    rel = str(md.relative_to(posts_dir))
    # 跳过 images 下的嵌入 md
    if "/images/" in rel:
        continue
    
    title_m = re.search(r'^title:\s*(.+)$', md.read_text(encoding='utf-8'), re.MULTILINE)
    title = title_m.group(1).strip().strip("'\"") if title_m else md.stem
    
    cover = get_cover(rel, title)
    if not cover:
        results["no_cover"].append(rel)
        continue
    
    changed = update_frontmatter(md, cover)
    key = f"{rel} -> {cover}"
    if changed:
        results["updated"].append(key)
    else:
        results["skipped"].append(key)

print(f"=== 更新结果 ===")
print(f"✓ 已更新: {len(results['updated'])} 篇")
print(f"- 无变化: {len(results['skipped'])} 篇")
print(f"⚠ 未匹配规则: {len(results['no_cover'])} 篇")
if results['no_cover']:
    print("\n未匹配的文章:")
    for r in results['no_cover']:
        print(f"  {r}")
print("\n已更新明细:")
for r in results['updated']:
    print(f"  {r}")
