# templatesearch-server(纯向量方案)

数据流:search.txt → SiliconFlow(bge-m3)批量向量化 → 内存向量索引;
查询词 → SiliconFlow 转向量 → 余弦相似度 → Top-N 模版 ID。
任意语言查询(中/英/日/西…)均可命中英文模版名。

## 运行

```bash
export SILICONFLOW_API_KEY=sk-你的key   # 必填,务必用重新生成的 key
# 国内站 key 需要:export SILICONFLOW_BASE=https://api.siliconflow.cn
go mod tidy
go run .
```

首次启动自动生成 vectors.jsonl(1313 条约 1 分钟,日志有进度),之后直接加载。

## 接口

```bash
# 检索:返回 [{id, name, score}],score 为余弦相似度
curl 'localhost:8080/api/templates/search?q=圣诞树&limit=10'

# search.txt 更新后:重新生成向量并热加载(并发保护)
curl -X POST localhost:8080/api/templates/reload

# 健康检查
curl localhost:8080/healthz
```

## 结构

- main.go                 gin 路由:查询向量化 + 向量索引检索
- internal/vectorize/     SiliconFlow 批量向量生成(限流重试、原子写文件)
- integration/            端到端集成测试(mock SiliconFlow)
- search.txt              模版数据(名称:ID)

## 注意

- 查询路径强依赖 SiliconFlow:embedding 调用失败时接口返回 503(无降级)
- 建议在网关/业务侧对热门查询做缓存(query 为 key),可大幅减少 embedding 调用量与延迟
- 模型固定 BAAI/bge-m3,生成与查询必须用同一模型

## 测试

```bash
go test ./...
```
