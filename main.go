// 模版相似检索服务(纯向量方案):
// search.txt -> SiliconFlow 批量向量化 -> 内存向量索引;
// 查询时同样经 SiliconFlow 转向量,余弦相似度返回 Top-N 模版 ID。
//
// 环境变量:
//
//	SILICONFLOW_API_KEY  必填;切勿写死在代码里
//	SILICONFLOW_BASE     默认 https://api.siliconflow.com
//	EMBED_MODEL          默认 BAAI/bge-m3(国际站请用 Qwen/Qwen3-Embedding-0.6B)
//	TEMPLATE_FILE        默认 search.txt
//	VECTOR_FILE          默认 vectors.jsonl
//	LISTEN_ADDR          默认 :8080
package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	templatesearch "github.com/testzhaoxiaofei/templatesearch"

	"templatesearch-server/internal/vectorize"
)

type app struct {
	vectors    *templatesearch.VectorIndex
	embedder   templatesearch.Embedder
	vecClient  *vectorize.Client
	dataFile   string
	vectorFile string
	rebuildMu  sync.Mutex
}

func main() {
	loadDotEnv(".env") // 自动加载当前目录 .env(已 export 的变量优先,不会被覆盖)

	apiKey := os.Getenv("SILICONFLOW_API_KEY")
	if apiKey == "" {
		log.Fatal("SILICONFLOW_API_KEY is required")
	}
	base := getenv("SILICONFLOW_BASE", "https://api.siliconflow.com")
	model := getenv("EMBED_MODEL", "BAAI/bge-m3")

	a := &app{
		dataFile:   getenv("TEMPLATE_FILE", "search.txt"),
		vectorFile: getenv("VECTOR_FILE", "vectors.jsonl"),
		vecClient:  vectorize.NewClient(base, model, apiKey),
		embedder:   templatesearch.NewHTTPEmbedder(base+"/v1/embeddings", model, apiKey),
		vectors:    templatesearch.NewVectorIndex(),
	}
	log.Printf("[startup] config: base=%s model=%s template_file=%s vector_file=%s api_key=%s",
		base, model, a.dataFile, a.vectorFile, maskKey(apiKey))

	// vectors.jsonl 不存在则启动时自动生成
	if _, err := os.Stat(a.vectorFile); os.IsNotExist(err) {
		log.Printf("[startup] %s not found, generating via SiliconFlow ...", a.vectorFile)
		genStart := time.Now()
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
		n, err := vectorize.GenerateJSONL(ctx, a.vecClient, a.dataFile, a.vectorFile, 32,
			func(done, total int) {
				log.Printf("[startup] embedding progress %d/%d (%.0f%%) elapsed=%s",
					done, total, float64(done)/float64(total)*100, time.Since(genStart).Round(time.Second))
			})
		cancel()
		if err != nil {
			log.Fatalf("[startup] generate vectors: %v", err)
		}
		log.Printf("[startup] generated %d vectors -> %s in %s", n, a.vectorFile, time.Since(genStart).Round(time.Millisecond))
	} else {
		log.Printf("[startup] found existing %s, skip generation", a.vectorFile)
	}

	loadStart := time.Now()
	if err := a.vectors.LoadJSONL(a.vectorFile); err != nil {
		log.Fatalf("[startup] load vectors from %s: %v", a.vectorFile, err)
	}
	log.Printf("[startup] vector index ready: %d templates, model=%s, load_cost=%s",
		a.vectors.Size(), model, time.Since(loadStart).Round(time.Millisecond))

	r := gin.Default()
	r.GET("/healthz", a.handleHealth)
	r.GET("/api/templates/search", a.handleSearch)
	r.POST("/api/templates/reload", a.handleReload)

	addr := getenv("LISTEN_ADDR", ":8080")
	log.Printf("listening on %s", addr)
	if err := r.Run(addr); err != nil {
		log.Fatal(err)
	}
}

func (a *app) handleHealth(c *gin.Context) {
	c.JSON(http.StatusOK, gin.H{"status": "ok", "vectors": a.vectors.Size()})
}

// GET /api/templates/search?q=圣诞树&limit=10
// 查询词 -> SiliconFlow 向量化 -> 余弦相似度 Top-N
func (a *app) handleSearch(c *gin.Context) {
	q := c.Query("q")
	if q == "" {
		c.JSON(http.StatusBadRequest, gin.H{"code": 400, "msg": "missing query param: q"})
		return
	}
	limit := 10
	if v := c.Query("limit"); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 && n <= 50 {
			limit = n
		}
	}

	ctx, cancel := context.WithTimeout(c.Request.Context(), 5*time.Second)
	defer cancel()

	embedStart := time.Now()
	qVec, err := a.embedder.Embed(ctx, q)
	embedCost := time.Since(embedStart)
	if err != nil {
		log.Printf("[search] q=%q embed FAILED cost=%s err=%v", q, embedCost.Round(time.Millisecond), err)
		c.JSON(http.StatusServiceUnavailable, gin.H{"code": 503, "msg": "embedding service error: " + err.Error()})
		return
	}

	searchStart := time.Now()
	results := a.vectors.Search(qVec, limit)
	searchCost := time.Since(searchStart)

	log.Printf("[search] q=%q limit=%d embed_cost=%s search_cost=%s results=%d top=%s",
		q, limit, embedCost.Round(time.Millisecond), searchCost.Round(time.Microsecond),
		len(results), topSummary(results, 3))

	c.JSON(http.StatusOK, gin.H{"code": 0, "query": q, "count": len(results), "results": results})
}

// POST /api/templates/reload
// search.txt 更新后:重新生成向量并热加载
func (a *app) handleReload(c *gin.Context) {
	if !a.rebuildMu.TryLock() {
		c.JSON(http.StatusConflict, gin.H{"code": 409, "msg": "rebuild already in progress"})
		return
	}
	defer a.rebuildMu.Unlock()

	log.Printf("[reload] start: regenerating vectors from %s", a.dataFile)
	start := time.Now()
	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Minute)
	defer cancel()
	n, err := vectorize.GenerateJSONL(ctx, a.vecClient, a.dataFile, a.vectorFile, 32,
		func(done, total int) { log.Printf("[reload] embedding progress %d/%d", done, total) })
	if err != nil {
		log.Printf("[reload] FAILED after %s: %v", time.Since(start).Round(time.Millisecond), err)
		c.JSON(http.StatusInternalServerError, gin.H{"code": 500, "msg": "regen vectors: " + err.Error()})
		return
	}
	if err := a.vectors.LoadJSONL(a.vectorFile); err != nil {
		log.Printf("[reload] load vectors FAILED: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"code": 500, "msg": "reload vectors: " + err.Error()})
		return
	}
	log.Printf("[reload] done: %d vectors in %s", n, time.Since(start).Round(time.Millisecond))
	c.JSON(http.StatusOK, gin.H{"code": 0, "vectors": n})
}

// maskKey 日志中对 API key 打码,只露前 6 后 4 位
func maskKey(k string) string {
	if len(k) <= 10 {
		return "***"
	}
	return k[:6] + "..." + k[len(k)-4:]
}

// topSummary 取前 n 条结果摘要用于日志
func topSummary(rs []templatesearch.Result, n int) string {
	if len(rs) == 0 {
		return "[]"
	}
	if n > len(rs) {
		n = len(rs)
	}
	s := "["
	for i := 0; i < n; i++ {
		if i > 0 {
			s += ", "
		}
		s += fmt.Sprintf("{id=%d name=%q score=%.4f}", rs[i].ID, rs[i].Name, rs[i].Score)
	}
	return s + "]"
}

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// loadDotEnv 读取 KEY=VALUE 格式的 .env 文件写入进程环境。
// 规则:文件不存在则静默跳过;# 开头为注释;已存在的环境变量不覆盖
// (即 export 的值优先于 .env);值两侧的引号会被去掉。
func loadDotEnv(path string) {
	data, err := os.ReadFile(path)
	if err != nil {
		return
	}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		line = strings.TrimPrefix(line, "export ")
		eq := strings.Index(line, "=")
		if eq <= 0 {
			continue
		}
		key := strings.TrimSpace(line[:eq])
		val := strings.TrimSpace(line[eq+1:])
		val = strings.Trim(val, "\"'")
		if key == "" || os.Getenv(key) != "" {
			continue
		}
		os.Setenv(key, val)
	}
	log.Printf("[startup] loaded env from %s", path)
}
