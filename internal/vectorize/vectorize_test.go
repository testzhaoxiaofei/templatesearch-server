package vectorize

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"sync/atomic"
	"testing"
)

// mockServer 模拟 OpenAI 兼容 embeddings 接口,前 failFirst 次返回 429
func mockServer(t *testing.T, failFirst int) (*httptest.Server, *int64) {
	var calls int64
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		n := atomic.AddInt64(&calls, 1)
		if int(n) <= failFirst {
			w.WriteHeader(http.StatusTooManyRequests)
			return
		}
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
		for i := range req.Input {
			out.Data = append(out.Data, d{Index: i, Embedding: []float32{float32(i), 1, 0}})
		}
		_ = json.NewEncoder(w).Encode(out)
	}))
	t.Cleanup(srv.Close)
	return srv, &calls
}

func TestGenerateJSONL(t *testing.T) {
	srv, _ := mockServer(t, 0)

	dir := t.TempDir()
	src := filepath.Join(dir, "search.txt")
	out := filepath.Join(dir, "vectors.jsonl")
	os.WriteFile(src, []byte("Christmas Tree:207\nwedding dress:122\n圣诞树:207\nbad line\n"), 0o644)

	c := NewClient(srv.URL, "BAAI/bge-m3", "test-key")
	n, err := GenerateJSONL(context.Background(), c, src, out, 2, nil)
	if err != nil {
		t.Fatal(err)
	}
	if n != 3 {
		t.Fatalf("want 3 records, got %d", n)
	}

	data, _ := os.ReadFile(out)
	lines := 0
	for _, b := range data {
		if b == '\n' {
			lines++
		}
	}
	if lines != 3 {
		t.Fatalf("want 3 jsonl lines, got %d", lines)
	}
	var rec struct {
		ID     int       `json:"id"`
		Name   string    `json:"name"`
		Vector []float32 `json:"vector"`
	}
	if err := json.Unmarshal(data[:indexByte(data, '\n')], &rec); err != nil {
		t.Fatal(err)
	}
	if rec.ID != 207 || rec.Name != "Christmas Tree" || len(rec.Vector) != 3 {
		t.Fatalf("bad first record: %+v", rec)
	}
}

func TestRetryOn429(t *testing.T) {
	srv, calls := mockServer(t, 2) // 前两次 429,第三次成功
	c := NewClient(srv.URL, "m", "k")
	vecs, err := c.EmbedBatch(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	if len(vecs) != 2 || *calls != 3 {
		t.Fatalf("want 2 vecs after 3 calls, got %d vecs %d calls", len(vecs), *calls)
	}
}

func TestNoRetryOn401(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	t.Cleanup(srv.Close)
	c := NewClient(srv.URL, "m", "bad-key")
	if _, err := c.EmbedBatch(context.Background(), []string{"a"}); err == nil {
		t.Fatal("expected error on 401")
	}
}

func indexByte(b []byte, c byte) int {
	for i, x := range b {
		if x == c {
			return i
		}
	}
	return len(b)
}
