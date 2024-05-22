# textrank_model.py

from konlpy.tag import Komoran
from textrank import KeywordSummarizer


class KomoranTokenizer:
    def __init__(self):
        self.komoran = Komoran()
        
    def tokenize(self, sent):
        words = self.komoran.pos(sent, join=True)
        words = [w for w in words if '/NN' in w]
        return words

def extract_filtered_words(keywords):
    filtered_keywords = []
    for words, rank in keywords:
        filtered_words = [word.split('/')[0] for word in words.split() if len(word.split('/')[0]) >= 2 and all(char >= '가' and char <= '힣' for char in word.split('/')[0])]
        if filtered_words:
            filtered_keywords.extend(filtered_words)
    return filtered_keywords[:10]  #10개만 출력되도록

def summarize_keywords(filename):
    sents = [sent.strip() for sent in filename.values() if sent.strip()]
    tokenizer = KomoranTokenizer()
    keyword_summarizer = KeywordSummarizer(tokenize=tokenizer.tokenize, min_count=2, min_cooccurrence=1)
    keywords = keyword_summarizer.summarize(sents, topk=20)
    return extract_filtered_words(keywords)

# def print_keywords(transcriptions):
#     summarized_keywords = summarize_keywords_from_list(transcriptions)
#     for keyword in summarized_keywords:
#         print(keyword)



