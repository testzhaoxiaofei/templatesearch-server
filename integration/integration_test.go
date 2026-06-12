// Package integration 用真实 search.txt 验证完整管线:
// 生成 vectors.jsonl -> 加载向量索引 -> 混合检索(SiliconFlow 用 mock 替代)。
package integration

import (
	"context"
	"encoding/json"
	"hash/fnv"
	"math"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	templatesearch "github.com/testzhaoxiaofei/templatesearch"

	"templatesearch-server/internal/vectorize"
)

// fakeEmbed 确定性伪向量:语义近似词映射到相近向量,
// 仅用于验证管线正确性,不代表真实 bge-m3 质量。
func fakeEmbed(text string) []float32 {
	// 简单跨语言词表:中文词与英文词共享同一语义桶
	synonym := map[string]string{"圣诞": "christmas", "树": "tree", "婚纱": "wedding dress"}
	t := strings.ToLower(text)
	for zh, en := range synonym {
		t = strings.ReplaceAll(t, zh, " "+en+" ")
	}
	vec := make([]float32, 64)
	for _, w := range strings.Fields(t) {
		h := fnv.New32a()
		h.Write([]byte(w))
		vec[h.Sum32()%64] += 1
	}
	var norm float64
	for _, x := range vec {
		norm += float64(x) * float64(x)
	}
	if norm > 0 {
		inv := float32(1 / math.Sqrt(norm))
		for i := range vec {
			vec[i] *= inv
		}
	}
	return vec
}

func mockSiliconFlow(t *testing.T) *httptest.Server {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req struct {
			Input []string `json:"input"`
		}
		_ = json.NewDecoder(r.Body).Decode(&req)
		type d struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		}
		out := struct {
			Data []d `json:"data"`
		}{}
		for i, text := range req.Input {
			out.Data = append(out.Data, d{Index: i, Embedding: fakeEmbed(text)})
		}
		_ = json.NewEncoder(w).Encode(out)
	}))
	t.Cleanup(srv.Close)
	return srv
}

type mockEmbedder struct{ srvURL string }

func (m *mockEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	return fakeEmbed(text), nil
}

func TestFullPipeline(t *testing.T) {
	srv := mockSiliconFlow(t)
	client := vectorize.NewClient(srv.URL, "BAAI/bge-m3", "test-key")

	// 1. 用真实的 search.txt 生成 vectors.jsonl
	out := filepath.Join(t.TempDir(), "vectors.jsonl")
	n, err := vectorize.GenerateJSONL(context.Background(), client, "../search.txt", out, 64, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("generated %d vectors", n)
	if n < 1300 {
		t.Fatalf("expected ~1313 vectors, got %d", n)
	}

	// 2. 加载向量索引
	vi := templatesearch.NewVectorIndex()
	if err := vi.LoadJSONL(out); err != nil {
		t.Fatal(err)
	}
	if vi.Size() != n {
		t.Fatalf("vector index size %d != generated %d", vi.Size(), n)
	}

	// 3. 纯向量检索:查询词转向量 -> 余弦 TopN
	emb := &mockEmbedder{}
	search := func(q string, topN int) []templatesearch.Result {
		qv, err := emb.Embed(context.Background(), q)
		if err != nil {
			t.Fatal(err)
		}
		return vi.Search(qv, topN)
	}

	rs := search("christmas tree", 10)
	if len(rs) != 10 || !strings.Contains(strings.ToLower(rs[0].Name), "christmas") {
		t.Fatalf("english query top1 = %+v", rs[0])
	}

	// 4. 跨语言:中文查询命中英文模版
	rs = search("圣诞树", 10)
	found := false
	for _, r := range rs[:3] {
		if strings.Contains(strings.ToLower(r.Name), "christmas") {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("cross-lingual query failed, top3: %v %v %v", rs[0], rs[1], rs[2])
	}
	t.Logf("cross-lingual top1: %+v", rs[0])
}
