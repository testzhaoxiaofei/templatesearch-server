// Package templatesearch 提供模版名称的相似度检索:
// 给定用户查询词,从模版库中返回最相近的 Top-N 模版及其 ID。
//
// 零外部依赖(仅标准库),万级数据内存索引、毫秒级查询,
// 内置多语言归一化(变音符折叠/全角转换/中日韩泰等无空格文种),
// 可选叠加多语言 embedding 向量检索实现跨语言语义匹配。
//
// 最简用法:
//
//	eng := templatesearch.NewEngine()
//	_ = eng.LoadFile("search.txt") // 每行 "名称:ID"
//	results := eng.Search("christmas tree", 10)
//	for _, r := range results {
//		fmt.Println(r.ID, r.Name, r.Score)
//	}
//
// 跨语言混合检索(词面 + 向量,RRF 融合,向量服务故障自动降级):
//
//	vi := templatesearch.NewVectorIndex()
//	_ = vi.LoadJSONL("vectors.jsonl") // tools/gen_vectors.py 离线生成
//	h := &templatesearch.Hybrid{
//		Lexical:  eng,
//		Vectors:  vi,
//		Embedder: templatesearch.NewHTTPEmbedder(embedURL, "bge-m3", apiKey),
//	}
//	results = h.Search(ctx, "圣诞树", 10)
package templatesearch
