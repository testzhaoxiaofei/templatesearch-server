package templatesearch

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"os"
	"sync"
	"time"
)

// Embedder 把一段文本变成向量。线上接多语言 embedding 服务
// (如自建 bge-m3 / multilingual-e5,或云端 embedding API)。
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
}

// VectorIndex 内存向量索引:1 万条 × 1024 维暴力余弦扫描仅需几毫秒,
// 无需引入向量数据库;到几十万条再考虑 HNSW / pgvector / Milvus。
type VectorIndex struct {
	mu   sync.RWMutex
	dim  int
	ids  []int
	name []string
	vecs [][]float32 // 已 L2 归一化,余弦相似度 = 点积
}

// NewVectorIndex 创建空索引
func NewVectorIndex() *VectorIndex { return &VectorIndex{} }

// vectorRecord 向量文件中的一行(JSONL)
type vectorRecord struct {
	ID     int       `json:"id"`
	Name   string    `json:"name"`
	Vector []float32 `json:"vector"`
}

// LoadJSONL 从 JSONL 文件加载离线算好的模版向量,
// 每行: {"id":207,"name":"Christmas Tree","vector":[0.01,...]}
func (v *VectorIndex) LoadJSONL(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	var ids []int
	var names []string
	var vecs [][]float32
	dim := 0

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<22)
	for sc.Scan() {
		line := bytes.TrimSpace(sc.Bytes())
		if len(line) == 0 {
			continue
		}
		var rec vectorRecord
		if err := json.Unmarshal(line, &rec); err != nil {
			return fmt.Errorf("parse vector line: %w", err)
		}
		if dim == 0 {
			dim = len(rec.Vector)
		} else if len(rec.Vector) != dim {
			return fmt.Errorf("inconsistent vector dim: %d vs %d (id=%d)", len(rec.Vector), dim, rec.ID)
		}
		l2normalize(rec.Vector)
		ids = append(ids, rec.ID)
		names = append(names, rec.Name)
		vecs = append(vecs, rec.Vector)
	}
	if err := sc.Err(); err != nil {
		return err
	}

	v.mu.Lock()
	v.dim, v.ids, v.name, v.vecs = dim, ids, names, vecs
	v.mu.Unlock()
	return nil
}

// Size 索引中的向量条数
func (v *VectorIndex) Size() int {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return len(v.ids)
}

// Search 用查询向量做余弦 TopN
func (v *VectorIndex) Search(qVec []float32, topN int) []Result {
	if topN <= 0 {
		topN = 10
	}
	v.mu.RLock()
	defer v.mu.RUnlock()
	if len(v.vecs) == 0 || len(qVec) != v.dim {
		return nil
	}
	q := make([]float32, len(qVec))
	copy(q, qVec)
	l2normalize(q)

	heap := make([]Result, 0, topN)
	for i, vec := range v.vecs {
		var dot float32
		for j := range vec {
			dot += vec[j] * q[j]
		}
		r := Result{ID: v.ids[i], Name: v.name[i], Score: float64(dot)}
		if len(heap) < topN {
			heap = append(heap, r)
			siftUp(heap, len(heap)-1)
		} else if less(heap[0], r) {
			heap[0] = r
			siftDown(heap, 0)
		}
	}
	sortDesc(heap)
	return heap
}

func l2normalize(v []float32) {
	var sum float64
	for _, x := range v {
		sum += float64(x) * float64(x)
	}
	if sum == 0 {
		return
	}
	inv := float32(1.0 / math.Sqrt(sum))
	for i := range v {
		v[i] *= inv
	}
}

// ---------- HTTP Embedder ----------

// HTTPEmbedder 调用兼容 OpenAI embeddings 协议的服务:
// POST {url}  body: {"model": "...", "input": "text"}
// resp: {"data":[{"embedding":[...]}]}
// 自建 TEI(text-embeddings-inference)+ bge-m3 即暴露该协议。
type HTTPEmbedder struct {
	URL    string
	Model  string
	APIKey string
	Client *http.Client
}

// NewHTTPEmbedder 创建带超时的 embedding 客户端
func NewHTTPEmbedder(url, model, apiKey string) *HTTPEmbedder {
	return &HTTPEmbedder{
		URL: url, Model: model, APIKey: apiKey,
		Client: &http.Client{Timeout: 2 * time.Second},
	}
}

// Embed 实现 Embedder
func (h *HTTPEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	body, _ := json.Marshal(map[string]any{"model": h.Model, "input": text})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, h.URL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if h.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+h.APIKey)
	}
	resp, err := h.Client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("embedding service status %d", resp.StatusCode)
	}
	var out struct {
		Data []struct {
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, err
	}
	if len(out.Data) == 0 {
		return nil, fmt.Errorf("empty embedding response")
	}
	return out.Data[0].Embedding, nil
}
