import MeCab
mecab = MeCab.Tagger("-Ochasen")

print(mecab.parse("平成最後の初売りセールが開催中"))