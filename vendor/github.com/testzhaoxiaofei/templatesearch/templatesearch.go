// Package search 实现模版名称的相似度检索。
//
// 思路:
//  1. 启动时把每个模版名归一化(小写、去标点),切成 n-gram(英文按词内 2-gram,
//     中文按单字 + 双字)建立倒排索引;
//  2. 查询时同样切 gram,通过倒排索引累计与每个候选模版的公共 gram 数,
//     用 Dice 系数 (2*公共/总量) 得到基础分;
//  3. 叠加业务加权:完全相等、前缀、包含、整词命中分别加分;
//  4. 取 Top-N 返回。
//
// 1000 量级的模版,内存索引 + 单机即可,QPS 轻松上万。
package templatesearch

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
	"sync"
	"unicode"
)

// Template 一条模版记录
type Template struct {
	ID   int    `json:"id"`
	Name string `json:"name"`
}

// Result 检索结果
type Result struct {
	ID    int     `json:"id"`
	Name  string  `json:"name"`
	Score float64 `json:"score"`
}

// posting 倒排表项:某 gram 在第 doc 个模版中出现 freq 次
type posting struct {
	doc  int32
	freq int32
}

// Engine 内存检索引擎(读多写少,RWMutex 保护,支持热更新)
type Engine struct {
	mu         sync.RWMutex
	templates  []Template
	normNames  []string             // 归一化后的名称,与 templates 下标对应
	wordSets   [][]string           // 每个模版的整词集合
	gramCount  []int                // 每个模版的 gram 总数
	inverted   map[string][]posting // gram -> (模版下标, 频次) 列表
	countsPool sync.Pool            // 复用查询计数数组
}

// NewEngine 创建空引擎
func NewEngine() *Engine {
	return &Engine{inverted: make(map[string][]posting)}
}

// LoadFile 从 "名称:ID" 格式的文件加载并重建索引
func (e *Engine) LoadFile(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return e.LoadReader(f)
}

// LoadReader 从任意 io.Reader 加载 "名称:ID" 格式数据并重建索引,
// 便于从 embed.FS、HTTP 响应、数据库导出流等来源加载。
func (e *Engine) LoadReader(r io.Reader) error {
	var ts []Template
	sc := bufio.NewScanner(r)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		idx := strings.LastIndex(line, ":")
		if idx <= 0 {
			continue
		}
		var id int
		if _, err := fmt.Sscanf(line[idx+1:], "%d", &id); err != nil {
			continue
		}
		ts = append(ts, Template{ID: id, Name: strings.TrimSpace(line[:idx])})
	}
	if err := sc.Err(); err != nil {
		return err
	}
	e.Rebuild(ts)
	return nil
}

// Rebuild 用给定模版列表重建索引(可用于从 DB 热加载)
func (e *Engine) Rebuild(ts []Template) {
	normNames := make([]string, len(ts))
	wordSets := make([][]string, len(ts))
	gramCount := make([]int, len(ts))
	inverted := make(map[string][]posting, len(ts)*8)

	for i, t := range ts {
		norm := normalize(t.Name)
		normNames[i] = norm
		wordSets[i] = words(norm)
		grams := ngrams(norm)
		gramCount[i] = len(grams)
		freq := make(map[string]int32, len(grams))
		for _, g := range grams {
			freq[g]++
		}
		for g, f := range freq {
			inverted[g] = append(inverted[g], posting{doc: int32(i), freq: f})
		}
	}

	e.mu.Lock()
	e.templates = ts
	e.normNames = normNames
	e.wordSets = wordSets
	e.gramCount = gramCount
	e.inverted = inverted
	e.mu.Unlock()
}

// Size 当前索引的模版数
func (e *Engine) Size() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return len(e.templates)
}

// Search 返回与 query 最相近的 topN 个模版,按分数降序
func (e *Engine) Search(query string, topN int) []Result {
	if topN <= 0 {
		topN = 10
	}
	qNorm := normalize(query)
	if qNorm == "" {
		return nil
	}
	qGrams := ngrams(qNorm)
	qWords := words(qNorm)

	e.mu.RLock()
	defer e.mu.RUnlock()

	// 1. 倒排索引累计公共 gram 数(平铺数组计数,避免 map 开销)
	counts := e.getCounts()
	defer e.putCounts(counts)
	touched := make([]int32, 0, 512)

	qGramFreq := make(map[string]int32, len(qGrams))
	for _, g := range qGrams {
		qGramFreq[g]++
	}
	for g, qf := range qGramFreq {
		for _, p := range e.inverted[g] {
			c := qf
			if p.freq < c {
				c = p.freq
			}
			if counts[p.doc] == 0 {
				touched = append(touched, p.doc)
			}
			counts[p.doc] += c
		}
	}

	// 2. 打分,小顶堆维护 TopN(避免对全部候选做整体排序)
	heap := make([]Result, 0, topN)
	for _, di32 := range touched {
		di := int(di32)
		c := counts[di32]
		dice := 0.0
		if total := len(qGrams) + e.gramCount[di]; total > 0 {
			dice = 2 * float64(c) / float64(total)
		}
		score := dice
		name := e.normNames[di]

		switch {
		case name == qNorm: // 完全相等
			score += 1.0
		case strings.HasPrefix(name, qNorm) || strings.HasPrefix(qNorm, name): // 前缀
			score += 0.35
		case strings.Contains(name, qNorm) || strings.Contains(qNorm, name): // 包含
			score += 0.25
		}
		// 整词命中:查询词在模版词集合中出现的比例
		if len(qWords) > 0 {
			hit := 0
			for _, qw := range qWords {
				for _, tw := range e.wordSets[di] {
					if qw == tw {
						hit++
						break
					}
				}
			}
			score += 0.4 * float64(hit) / float64(len(qWords))
		}

		r := Result{ID: e.templates[di].ID, Name: e.templates[di].Name, Score: score}
		if len(heap) < topN {
			heap = append(heap, r)
			siftUp(heap, len(heap)-1)
		} else if less(heap[0], r) {
			heap[0] = r
			siftDown(heap, 0)
		}
	}

	// 3. 堆内元素按分数降序输出(分数相同按 ID 升序,保证结果稳定)
	sortDesc(heap)
	return heap
}

// sortDesc 按分数降序、同分按 ID 升序排序
func sortDesc(rs []Result) {
	sort.Slice(rs, func(i, j int) bool { return less(rs[j], rs[i]) })
}

// less 排序优先级:分数小的"更小";同分时 ID 大的"更小"(这样堆顶始终是最该被淘汰的)
func less(a, b Result) bool {
	if a.Score != b.Score {
		return a.Score < b.Score
	}
	return a.ID > b.ID
}

func siftUp(h []Result, i int) {
	for i > 0 {
		p := (i - 1) / 2
		if !less(h[i], h[p]) {
			break
		}
		h[i], h[p] = h[p], h[i]
		i = p
	}
}

func siftDown(h []Result, i int) {
	n := len(h)
	for {
		l, r, s := 2*i+1, 2*i+2, i
		if l < n && less(h[l], h[s]) {
			s = l
		}
		if r < n && less(h[r], h[s]) {
			s = r
		}
		if s == i {
			return
		}
		h[i], h[s] = h[s], h[i]
		i = s
	}
}

// getCounts/putCounts 复用与模版数等长的计数数组,避免每次查询分配
func (e *Engine) getCounts() []int32 {
	if v := e.countsPool.Get(); v != nil {
		if s, ok := v.([]int32); ok && len(s) >= len(e.templates) {
			return s
		}
	}
	return make([]int32, len(e.templates))
}

func (e *Engine) putCounts(s []int32) {
	for i := range s {
		s[i] = 0
	}
	e.countsPool.Put(s) //nolint:staticcheck
}

// ---------- 文本处理 ----------

// normalize 小写化,标点/符号统一替换为空格,压缩空白
func normalize(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range strings.ToLower(s) {
		for _, fr := range foldRune(r) {
			switch {
			case unicode.IsLetter(fr) || unicode.IsDigit(fr):
				b.WriteRune(fr)
			default:
				b.WriteRune(' ')
			}
		}
	}
	return strings.Join(strings.Fields(b.String()), " ")
}

// words 按空格切词;中文整段作为一个"词"再拆单字
func words(norm string) []string {
	var out []string
	for _, w := range strings.Fields(norm) {
		if isCJKWord(w) {
			for _, r := range w {
				out = append(out, string(r))
			}
		} else {
			out = append(out, w)
		}
	}
	return out
}

// ngrams 生成 gram 列表:
//   - ASCII 词:整词 + 词内 2-gram(短词只保留整词)
//   - CJK 词:单字 + 相邻双字
func ngrams(norm string) []string {
	var grams []string
	for _, w := range strings.Fields(norm) {
		if isCJKWord(w) {
			rs := []rune(w)
			for i, r := range rs {
				grams = append(grams, string(r))
				if i+1 < len(rs) {
					grams = append(grams, string(rs[i:i+2]))
				}
			}
			continue
		}
		grams = append(grams, "w:"+w) // 整词带前缀,避免与 2-gram 撞 key
		rs := []rune(w)
		if len(rs) < 3 {
			continue
		}
		for i := 0; i+2 <= len(rs); i++ {
			grams = append(grams, string(rs[i:i+2]))
		}
	}
	return grams
}

func isCJKWord(w string) bool {
	for _, r := range w {
		if isNoSpaceScript(r) {
			return true
		}
	}
	return false
}
