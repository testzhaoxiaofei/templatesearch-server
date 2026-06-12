// Package vectorize 调用 SiliconFlow(OpenAI 兼容协议)批量生成模版向量,
// 输出 vectors.jsonl 供 templatesearch.VectorIndex 加载。
package vectorize

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

// Client OpenAI 兼容 embeddings 客户端
type Client struct {
	BaseURL string // 例如 https://api.siliconflow.com
	Model   string // 例如 BAAI/bge-m3
	APIKey  string
	HTTP    *http.Client
}

// NewClient 创建客户端
func NewClient(baseURL, model, apiKey string) *Client {
	return &Client{
		BaseURL: strings.TrimRight(baseURL, "/"),
		Model:   model,
		APIKey:  apiKey,
		HTTP:    &http.Client{Timeout: 60 * time.Second},
	}
}

// EmbedBatch 一次请求向量化多条文本,带指数退避重试(限流/瞬时故障)
func (c *Client) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	body, _ := json.Marshal(map[string]any{"model": c.Model, "input": texts})
	var lastErr error
	for attempt := 0; attempt < 5; attempt++ {
		if attempt > 0 {
			select {
			case <-time.After(time.Duration(1<<attempt) * time.Second):
			case <-ctx.Done():
				return nil, ctx.Err()
			}
		}
		vecs, retryable, err := c.doRequest(ctx, body, len(texts))
		if err == nil {
			return vecs, nil
		}
		lastErr = err
		if !retryable {
			log.Printf("[vectorize] embed failed (non-retryable): %v", err)
			return nil, err
		}
		log.Printf("[vectorize] embed attempt %d/5 failed, will retry: %v", attempt+1, err)
	}
	return nil, fmt.Errorf("embed failed after retries: %w", lastErr)
}

func (c *Client) doRequest(ctx context.Context, body []byte, want int) (vecs [][]float32, retryable bool, err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		c.BaseURL+"/v1/embeddings", bytes.NewReader(body))
	if err != nil {
		return nil, false, err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.APIKey)

	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, true, err // 网络错误可重试
	}
	defer resp.Body.Close()

	switch {
	case resp.StatusCode == http.StatusOK:
	case resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= 500:
		return nil, true, fmt.Errorf("status %d", resp.StatusCode) // 限流/服务端错误可重试
	default:
		var eb bytes.Buffer
		_, _ = eb.ReadFrom(resp.Body)
		return nil, false, fmt.Errorf("status %d: %s", resp.StatusCode, eb.String()) // 4xx 直接失败
	}

	var out struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, false, err
	}
	if len(out.Data) != want {
		return nil, false, fmt.Errorf("expected %d embeddings, got %d", want, len(out.Data))
	}
	vecs = make([][]float32, want)
	for _, d := range out.Data {
		if d.Index < 0 || d.Index >= want {
			return nil, false, fmt.Errorf("bad index %d in response", d.Index)
		}
		vecs[d.Index] = d.Embedding
	}
	return vecs, false, nil
}

// item 一条模版
type item struct {
	ID   int
	Name string
}

// parseTemplates 解析 "名称:ID" 文件
func parseTemplates(path string) ([]item, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var items []item
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		idx := strings.LastIndex(line, ":")
		if idx <= 0 {
			continue
		}
		id, err := strconv.Atoi(strings.TrimSpace(line[idx+1:]))
		if err != nil {
			continue
		}
		items = append(items, item{ID: id, Name: strings.TrimSpace(line[:idx])})
	}
	return items, sc.Err()
}

// GenerateJSONL 全量生成:读 templatesPath,批量向量化,原子写入 outPath。
// batchSize 建议 32;progress 非 nil 时回调进度(done/total)。
func GenerateJSONL(ctx context.Context, c *Client, templatesPath, outPath string,
	batchSize int, progress func(done, total int)) (int, error) {

	if batchSize <= 0 {
		batchSize = 32
	}
	items, err := parseTemplates(templatesPath)
	if err != nil {
		return 0, err
	}
	if len(items) == 0 {
		return 0, fmt.Errorf("no valid templates in %s", templatesPath)
	}

	tmp := outPath + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return 0, err
	}
	w := bufio.NewWriterSize(f, 1<<20)
	enc := json.NewEncoder(w)

	type record struct {
		ID     int       `json:"id"`
		Name   string    `json:"name"`
		Vector []float32 `json:"vector"`
	}

	for start := 0; start < len(items); start += batchSize {
		end := start + batchSize
		if end > len(items) {
			end = len(items)
		}
		chunk := items[start:end]
		texts := make([]string, len(chunk))
		for i, it := range chunk {
			texts[i] = it.Name
		}
		vecs, err := c.EmbedBatch(ctx, texts)
		if err != nil {
			f.Close()
			os.Remove(tmp)
			return 0, fmt.Errorf("batch %d-%d: %w", start, end, err)
		}
		for i, it := range chunk {
			if err := enc.Encode(record{ID: it.ID, Name: it.Name, Vector: vecs[i]}); err != nil {
				f.Close()
				os.Remove(tmp)
				return 0, err
			}
		}
		if progress != nil {
			progress(end, len(items))
		}
	}

	if err := w.Flush(); err != nil {
		f.Close()
		os.Remove(tmp)
		return 0, err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return 0, err
	}
	// 原子替换,服务读到的 vectors.jsonl 永远是完整文件
	if err := os.Rename(tmp, outPath); err != nil {
		os.Remove(tmp)
		return 0, err
	}
	return len(items), nil
}
