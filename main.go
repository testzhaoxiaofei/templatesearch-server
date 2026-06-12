// 模版相似检索服务(纯向量方案):
// search.txt -> SiliconFlow 批量向量化 -> 内存向量索引;
// 查询时同样经 SiliconFlow 转向量,余弦相似度返回 Top-N 模版 ID。
//
// 环境变量:
//
//	SILICONFLOW_API_KEY  必填;切勿写死在代码里
//	SILICONFLOW_BASE     默认 https://api.siliconflow.com
//	EMBED_MODEL          默认 BAAI/bge-m3
//	TEMPLATE_FILE        默认 search.txt
//	VECTOR_FILE          默认 vectors.jsonl
//	LISTEN_ADDR          默认 :8080
package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"strconv"
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

	// vectors.jsonl 不存在则启动时自动生成
	if _, err := os.Stat(a.vectorFile); os.IsNotExist(err) {
		log.Printf("%s not found, generating via SiliconFlow ...", a.vectorFile)
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Minute)
		n, err := vectorize.GenerateJSONL(ctx, a.vecClient, a.dataFile, a.vectorFile, 32,
			func(done, total int) { log.Printf("  embedding %d/%d", done, total) })
		cancel()
		if err != nil {
			log.Fatalf("generate vectors: %v", err)
		}
		log.Printf("generated %d vectors -> %s", n, a.vectorFile)
	}
	if err := a.vectors.LoadJSONL(a.vectorFile); err != nil {
		log.Fatalf("load vectors from %s: %v", a.vectorFile, err)
	}
	log.Printf("vector index ready: %d templates, model=%s", a.vectors.Size(), model)

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
	qVec, err := a.embedder.Embed(ctx, q)
	if err != nil {
		c.JSON(http.StatusServiceUnavailable, gin.H{"code": 503, "msg": "embedding service error: " + err.Error()})
		return
	}

	results := a.vectors.Search(qVec, limit)
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

	ctx, cancel := context.WithTimeout(c.Request.Context(), 10*time.Minute)
	defer cancel()
	n, err := vectorize.GenerateJSONL(ctx, a.vecClient, a.dataFile, a.vectorFile, 32, nil)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"code": 500, "msg": "regen vectors: " + err.Error()})
		return
	}
	if err := a.vectors.LoadJSONL(a.vectorFile); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"code": 500, "msg": "reload vectors: " + err.Error()})
		return
	}
	c.JSON(http.StatusOK, gin.H{"code": 0, "vectors": n})
}

func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}
