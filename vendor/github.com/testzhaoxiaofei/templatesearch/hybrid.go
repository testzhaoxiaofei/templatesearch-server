package templatesearch

import (
	"context"
	"log"
	"time"
)

// Hybrid 混合检索:词面引擎(精确/前缀/拼写容错)+ 向量引擎(跨语言语义),
// 用 RRF(Reciprocal Rank Fusion)融合两路排名。
// 向量服务超时或故障时自动降级为纯词面检索,保证可用性。
type Hybrid struct {
	Lexical  *Engine
	Vectors  *VectorIndex // 可为 nil
	Embedder Embedder     // 可为 nil
	// EmbedTimeout 查询向量化的超时上限,超过即降级,默认 300ms
	EmbedTimeout time.Duration
}

const rrfK = 60 // RRF 平滑常数,经验值

// Search 返回融合后的 TopN
func (h *Hybrid) Search(ctx context.Context, query string, topN int) []Result {
	if topN <= 0 {
		topN = 10
	}
	// 两路各取 3 倍候选再融合,提高召回
	recall := topN * 3

	lex := h.Lexical.Search(query, recall)

	var vec []Result
	if h.Vectors != nil && h.Embedder != nil && h.Vectors.Size() > 0 {
		tctx, cancel := context.WithTimeout(ctx, h.embedTimeout())
		defer cancel()
		if qv, err := h.Embedder.Embed(tctx, query); err == nil {
			vec = h.Vectors.Search(qv, recall)
		} else {
			log.Printf("embed degraded to lexical-only: %v", err) // 降级,不影响请求
		}
	}

	if len(vec) == 0 {
		if len(lex) > topN {
			lex = lex[:topN]
		}
		return lex
	}

	// RRF 融合:score = Σ 1/(k + rank)
	type agg struct {
		r     Result
		score float64
	}
	fused := make(map[int]*agg, len(lex)+len(vec))
	addList := func(list []Result, weight float64) {
		for rank, r := range list {
			s := weight / float64(rrfK+rank+1)
			if a, ok := fused[r.ID]; ok {
				a.score += s
			} else {
				fused[r.ID] = &agg{r: r, score: s}
			}
		}
	}
	addList(lex, 1.0)
	addList(vec, 1.0)

	out := make([]Result, 0, len(fused))
	for _, a := range fused {
		a.r.Score = a.score
		out = append(out, a.r)
	}
	sortDesc(out)
	if len(out) > topN {
		out = out[:topN]
	}
	return out
}

func (h *Hybrid) embedTimeout() time.Duration {
	if h.EmbedTimeout > 0 {
		return h.EmbedTimeout
	}
	return 300 * time.Millisecond
}
