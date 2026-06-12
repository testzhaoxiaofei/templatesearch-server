package templatesearch

import "unicode"

// foldRune 多语言归一化折叠:
//   - 全角字符 -> 半角(ＡＢＣ -> abc,全角数字/标点同理)
//   - 拉丁变音符 -> 基础字母(café -> cafe, niño -> nino, schön -> schon)
//
// 覆盖 Latin-1 Supplement 和 Latin Extended-A 常用区,
// 足以处理西/法/德/葡/意/土/北欧等主要拉丁语系输入。
func foldRune(r rune) []rune {
	// 全角 ASCII 区 -> 半角
	if r >= 0xFF01 && r <= 0xFF5E {
		return []rune{r - 0xFEE0}
	}
	if r == 0x3000 { // 全角空格
		return []rune{' '}
	}
	if m, ok := latinFold[r]; ok {
		return m
	}
	return []rune{r}
}

var latinFold = map[rune][]rune{
	'à': {'a'}, 'á': {'a'}, 'â': {'a'}, 'ã': {'a'}, 'ä': {'a'}, 'å': {'a'}, 'ā': {'a'}, 'ă': {'a'}, 'ą': {'a'},
	'è': {'e'}, 'é': {'e'}, 'ê': {'e'}, 'ë': {'e'}, 'ē': {'e'}, 'ĕ': {'e'}, 'ė': {'e'}, 'ę': {'e'}, 'ě': {'e'},
	'ì': {'i'}, 'í': {'i'}, 'î': {'i'}, 'ï': {'i'}, 'ĩ': {'i'}, 'ī': {'i'}, 'ĭ': {'i'}, 'į': {'i'}, 'ı': {'i'},
	'ò': {'o'}, 'ó': {'o'}, 'ô': {'o'}, 'õ': {'o'}, 'ö': {'o'}, 'ø': {'o'}, 'ō': {'o'}, 'ŏ': {'o'}, 'ő': {'o'},
	'ù': {'u'}, 'ú': {'u'}, 'û': {'u'}, 'ü': {'u'}, 'ũ': {'u'}, 'ū': {'u'}, 'ŭ': {'u'}, 'ů': {'u'}, 'ű': {'u'}, 'ų': {'u'},
	'ý': {'y'}, 'ÿ': {'y'}, 'ŷ': {'y'},
	'ç': {'c'}, 'ć': {'c'}, 'ĉ': {'c'}, 'ċ': {'c'}, 'č': {'c'},
	'ñ': {'n'}, 'ń': {'n'}, 'ņ': {'n'}, 'ň': {'n'},
	'ś': {'s'}, 'ŝ': {'s'}, 'ş': {'s'}, 'š': {'s'}, 'ß': {'s', 's'},
	'ź': {'z'}, 'ż': {'z'}, 'ž': {'z'},
	'ð': {'d'}, 'ď': {'d'}, 'đ': {'d'},
	'ĝ': {'g'}, 'ğ': {'g'}, 'ġ': {'g'}, 'ģ': {'g'},
	'ĥ': {'h'}, 'ħ': {'h'},
	'ĵ': {'j'},
	'ķ': {'k'},
	'ĺ': {'l'}, 'ļ': {'l'}, 'ľ': {'l'}, 'ł': {'l'},
	'ŕ': {'r'}, 'ŗ': {'r'}, 'ř': {'r'},
	'ţ': {'t'}, 'ť': {'t'}, 'ŧ': {'t'},
	'ŵ': {'w'},
	'æ': {'a', 'e'}, 'œ': {'o', 'e'}, 'þ': {'t', 'h'},
}

// isNoSpaceScript 判断字符是否属于"词间无空格"的文种,
// 这类文字按 字符 n-gram 切分而非按空格分词。
// 覆盖:中日韩、泰文、老挝文、高棉文、缅甸文。
func isNoSpaceScript(r rune) bool {
	return unicode.Is(unicode.Han, r) ||
		unicode.Is(unicode.Hiragana, r) ||
		unicode.Is(unicode.Katakana, r) ||
		unicode.Is(unicode.Hangul, r) ||
		unicode.Is(unicode.Thai, r) ||
		unicode.Is(unicode.Lao, r) ||
		unicode.Is(unicode.Khmer, r) ||
		unicode.Is(unicode.Myanmar, r)
}
