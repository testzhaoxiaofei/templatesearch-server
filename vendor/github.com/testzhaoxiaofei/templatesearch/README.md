# templatesearch

模版名称相似度检索 Go 库:给定用户查询词,返回最相近的 Top-N 模版及其 ID。

- **零外部依赖**,仅 Go 标准库,`go get` 即用
- 万级模版内存索引,单次查询 < 1ms(1 万条实测 0.68ms)
- 多语言归一化内置:变音符折叠(café→cafe)、全角转半角、中/日/韩/泰/老/高棉/缅甸文按字符 n-gram
- 拼写容错(`wedding dres` 命中 `wedding dress`)、精确/前缀/包含加权
- 可选多语言 embedding 混合检索(跨语言语义,RRF 融合,故障自动降级)
- 并发安全,支持热重载索引

## 安装

```bash
go get github.com/testzhaoxiaofei/templatesearch
```

## 快速开始

数据格式:每行 `名称:ID`

```
Christmas Tree:207
wedding dress:122
Birthday Party:570
```

```go
package main

import (
	"fmt"

	templatesearch "github.com/testzhaoxiaofei/templatesearch"
)

func main() {
	eng := templatesearch.NewEngine()
	if err := eng.LoadFile("search.txt"); err != nil {
		panic(err)
	}

	for _, r := range eng.Search("christmas tree", 10) {
		fmt.Printf("id=%d name=%s score=%.3f\n", r.ID, r.Name, r.Score)
	}
}
```

也可从任意来源加载:

```go
eng.LoadReader(strings.NewReader(data))      // 字符串 / HTTP / embed.FS
eng.Rebuild([]templatesearch.Template{...})  // 直接从数据库行构建
```

## 跨语言混合检索(可选)

中文/日语/西语等查询命中英文模版名,需要叠加多语言 embedding:

```go
vi := templatesearch.NewVectorIndex()
_ = vi.LoadJSONL("vectors.jsonl") // tools/gen_vectors.py 离线生成

h := &templatesearch.Hybrid{
	Lexical:  eng,
	Vectors:  vi,
	Embedder: templatesearch.NewHTTPEmbedder(
		"http://embed-svc:8081/v1/embeddings", "bge-m3", ""),
}
results := h.Search(ctx, "圣诞树", 10) // 命中 Christmas Tree
```

embedding 服务超时(默认 300ms,可调 `EmbedTimeout`)或故障时自动降级为纯词面检索。
零成本替代方案:把翻译好的别名直接追加进数据文件(`圣诞树:207`),引擎天然支持一个 ID 多个名称。

## 在你的项目中使用

本库是纯函数库,不含任何 HTTP 服务。在你自己的 gin/echo/grpc 服务里持有一个
`*Engine`(或 `*Hybrid`)单例,在 handler 里调用 `Search` 即可,引擎并发安全:

```go
var eng = templatesearch.NewEngine() // 启动时 LoadFile 一次,全局复用

func searchHandler(c *gin.Context) {
	results := eng.Search(c.Query("q"), 10)
	c.JSON(200, results)
}
```

数据更新后在任意 goroutine 调 `eng.LoadFile(...)` 即热重载,无需加锁、无需重启。

## API 一览

| 类型/函数 | 说明 |
|---|---|
| `NewEngine() *Engine` | 创建词面检索引擎 |
| `(*Engine) LoadFile / LoadReader / Rebuild` | 加载数据并重建索引(可热重载) |
| `(*Engine) Search(query string, topN int) []Result` | 检索,返回按分数降序的 `{ID, Name, Score}` |
| `NewVectorIndex() *VectorIndex` | 创建向量索引 |
| `(*VectorIndex) LoadJSONL(path)` | 加载离线向量(`{"id":..,"name":..,"vector":[..]}` 每行一条) |
| `Hybrid` | 词面 + 向量混合检索,RRF 融合 |
| `Embedder` 接口 / `NewHTTPEmbedder` | 查询向量化,兼容 OpenAI embeddings 协议(TEI 等) |

## 算法说明

归一化(小写、折叠、去标点)→ 切 gram(英文整词 + 词内 2-gram,无空格文种单字 + 双字)→
倒排索引累计公共 gram → Dice 系数基础分 + 精确(+1.0)/前缀(+0.35)/包含(+0.25)/整词命中(×0.4)加权 →
小顶堆取 Top-N。查询计数数组经 sync.Pool 复用,无每查询大分配。

## 版本发布

模块路径已固定为 `github.com/testzhaoxiaofei/templatesearch`。每次更新后:

```bash
git add . && git commit -m "..."
git push
git tag v0.1.1 && git push origin v0.1.1   # 递增版本号
```

使用方按版本号拉取(首次或代理未同步时可加 GOPROXY=direct):

```bash
go get github.com/testzhaoxiaofei/templatesearch@v0.1.1
```

## 测试

```bash
go test -race ./...
go test -bench . ./...
```

## License

MIT
