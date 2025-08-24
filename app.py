import streamlit as st
import pandas as pd
import numpy as np
import json
import re
import random
import os
from typing import Dict, List, Optional, Tuple
from collections import Counter
import math
import time
import requests

# 設定
st.set_page_config(
    page_title="AI大谷 - 改善版",
    layout="wide"
)

# 軽量テキスト検索クラス（scikit-learn不要）
class LightweightTextSearch:
    """軽量TF-IDF検索システム"""
    
    def __init__(self, texts: List[str], max_features: int = 2000):
        self.texts = texts
        self.max_features = max_features
        self.vocab = self._build_vocabulary()
        self.idf_vector = self._compute_idf()
    
    def _tokenize(self, text: str) -> List[str]:
        """日本語対応トークナイズ"""
        if not isinstance(text, str):
            return []
        tokens = re.findall(r'[\w\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+', text.lower())
        return [token for token in tokens if len(token) > 1]
    
    def _build_vocabulary(self) -> Dict[str, int]:
        """語彙辞書構築"""
        all_tokens = []
        for text in self.texts:
            all_tokens.extend(self._tokenize(text))
        
        token_counts = Counter(all_tokens)
        top_tokens = [token for token, count in token_counts.most_common(self.max_features)]
        
        return {token: idx for idx, token in enumerate(top_tokens)}
    
    def _compute_tf(self, tokens: List[str]) -> np.ndarray:
        """TF計算"""
        tf_vector = np.zeros(len(self.vocab))
        if not tokens:
            return tf_vector
            
        token_counts = Counter(tokens)
        total_tokens = len(tokens)
        
        for token, count in token_counts.items():
            if token in self.vocab:
                tf_vector[self.vocab[token]] = count / total_tokens
        
        return tf_vector
    
    def _compute_idf(self) -> np.ndarray:
        """IDF計算"""
        idf_vector = np.zeros(len(self.vocab))
        num_docs = len(self.texts)
        
        for token, token_idx in self.vocab.items():
            doc_count = sum(1 for text in self.texts if token in self._tokenize(text))
            if doc_count > 0:
                idf_vector[token_idx] = math.log(num_docs / doc_count)
        
        return idf_vector
    
    def _text_to_tfidf(self, text: str) -> np.ndarray:
        """テキストをTF-IDFベクトルに変換"""
        tokens = self._tokenize(text)
        tf_vector = self._compute_tf(tokens)
        return tf_vector * self.idf_vector
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """コサイン類似度"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """類似テキスト検索"""
        query_tfidf = self._text_to_tfidf(query)
        similarities = []
        
        for i, text in enumerate(self.texts):
            text_tfidf = self._text_to_tfidf(text)
            similarity = self.cosine_similarity(query_tfidf, text_tfidf)
            similarities.append((i, similarity))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

# 超軽量キーワード検索
class KeywordSearch:
    """軽量キーワードマッチング検索"""
    
    def __init__(self, texts: List[str]):
        self.texts = texts
        self.keyword_index = self._build_index()
    
    def _tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str):
            return []
        return re.findall(r'[\w\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+', text.lower())
    
    def _build_index(self) -> Dict[str, List[int]]:
        index = {}
        for doc_id, text in enumerate(self.texts):
            tokens = self._tokenize(text)
            for token in set(tokens):
                if len(token) > 1:
                    if token not in index:
                        index[token] = []
                    index[token].append(doc_id)
        return index
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        query_tokens = set(self._tokenize(query))
        doc_scores = {}
        
        for token in query_tokens:
            if token in self.keyword_index:
                for doc_id in self.keyword_index[token]:
                    doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1
        
        if not doc_scores:
            return []
        
        max_score = max(doc_scores.values())
        results = [(doc_id, score/max_score) for doc_id, score in doc_scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]

# メインRAGシステム
class ImprovedOhtaniRAG:
    """改善版大谷翔平RAGシステム"""
    
    def __init__(self, csv_path: str):
        self.df = self._load_data(csv_path)
        self.questions = self.df['Question'].fillna('').astype(str).tolist()
        self.answers = self.df['Answer'].fillna('').astype(str).tolist()
        
        # 検索システム初期化
        self.tfidf_search = LightweightTextSearch(self.questions)
        self.keyword_search = KeywordSearch(self.questions)
        self.answer_search = KeywordSearch(self.answers)
        
        # 大谷選手の話し方パターン（大幅拡張）
        self.ohtani_patterns = self._extract_speech_patterns()
        
        # トピック別回答テンプレート
        self.topic_templates = self._create_topic_templates()
    
    def _load_data(self, csv_path: str) -> pd.DataFrame:
        """データ読み込み"""
        try:
            df = pd.read_csv(csv_path)
            if len(df) < 50:
                parent_path = os.path.join(os.path.dirname(os.path.dirname(csv_path)), "ohtani_rag_final.csv")
                if os.path.exists(parent_path):
                    df = pd.read_csv(parent_path)
            return df
        except FileNotFoundError:
            return self._create_expanded_sample_data()
    
    def _create_expanded_sample_data(self) -> pd.DataFrame:
        """拡張サンプルデータ（100件以上）"""
        questions = [
            # 野球関連
            '今シーズンの目標は何ですか？',
            'バッティングで心がけていることは？',
            'ピッチングで大切にしていることは？',
            'チームの雰囲気はいかがですか？',
            'プレッシャーを感じる時はありますか？',
            'スランプの時はどう対処しますか？',
            '好きな球場はありますか？',
            '印象に残っている試合は？',
            'ホームランを打った時の気持ちは？',
            '完全試合に近づいた時の心境は？',
            
            # プライベート・趣味
            '趣味は何ですか？',
            '好きな音楽はありますか？',
            '愛用している物はありますか？',
            '休日の過ごし方は？',
            '好きな映画はありますか？',
            'ゲームはしますか？',
            '料理は作りますか？',
            'ペットは飼っていますか？',
            '読書はしますか？',
            '車は好きですか？',
            
            # 食べ物関連
            '好きな日本料理は？',
            'アメリカの食べ物で好きなものは？',
            '苦手な食べ物はありますか？',
            '試合前に食べるものは？',
            'ホームシックになった時の食べ物は？',
            '母親の手料理で好きなものは？',
            'アメリカで驚いた食べ物は？',
            '健康のために気をつけている食事は？',
            
            # トレーニング・健康
            'トレーニングメニューについて教えてください',
            '体調管理で気をつけていることは？',
            'けがを予防するために何をしていますか？',
            '筋トレで重視していることは？',
            '睡眠時間はどのくらいですか？',
            'ストレッチは毎日していますか？',
            '疲労回復の方法は？',
            'メンタルトレーニングはしていますか？',
            
            # 人間関係・コミュニケーション
            'チームメイトとはどう接していますか？',
            'コーチとの関係について教えてください',
            'ファンとの交流で印象的なことは？',
            '通訳の方との関係は？',
            '家族との連絡頻度は？',
            '友人とは連絡を取り合っていますか？',
            'メディア対応で心がけていることは？',
            
            # 将来・目標
            '10年後の自分をどう想像しますか？',
            '引退後は何をしたいですか？',
            '子供たちに伝えたいことは？',
            '野球界に貢献したいことは？',
            '日本球界復帰の可能性は？',
            'コーチになりたいですか？',
            
            # 文化・比較
            '日本とアメリカの野球の違いは？',
            'アメリカ生活で困ったことは？',
            '日本が恋しくなる時は？',
            'アメリカで学んだことは？',
            '両国の良いところは？',
            '言葉の壁はありましたか？',
            
            # 哲学・価値観
            '人生で大切にしていることは？',
            '困難を乗り越える秘訣は？',
            '成功の秘訣は何だと思いますか？',
            '感謝していることは？',
            '挑戦することの意味は？',
            'プロとして心がけていることは？'
        ]
        
        answers = [
            # 野球関連の回答
            'そうですね、今シーズンはチーム一丸となってワールドシリーズ制覇を目指したいと思います。個人的にも、投打両方でチームに貢献できるよう、日々努力を続けています。',
            'バッティングでは、相手投手をしっかりと研究して、状況に応じたアプローチを心がけています。特に、チャンスの場面では冷静さを保つことを大切にしています。',
            'ピッチングでは、コントロールを第一に考えています。ストライクゾーンでの勝負を意識しながら、バッターとの駆け引きを楽しんでいます。',
            'チームの雰囲気は本当に素晴らしいです。みんなが同じ目標に向かって努力していて、お互いを高め合える関係を築けています。',
            'プレッシャーは感じますが、それを楽しめるようになりました。大舞台でプレーできることに感謝して、リラックスして臨んでいます。',
            'スランプの時は、基本に立ち戻ることを心がけています。焦らずに、一つ一つの動作を丁寧に確認し直すようにしています。',
            'どの球場も特色があって素晴らしいですが、やはりホーム球場でプレーする時は特別な気持ちになりますね。ファンの皆さんの声援が力になります。',
            '初めてメジャーでホームランを打った試合は今でも鮮明に覚えています。夢が現実になった瞬間でした。',
            'ホームランを打った瞬間は、やはり嬉しいですね。でも、それよりもチームの勝利に貢献できたことが一番嬉しいです。',
            '完全試合に近づいた時は、集中力を切らさないよう注意していました。一球一球に集中することだけを考えていました。',
            
            # プライベート・趣味の回答
            'オフの時間は映画を見たり、音楽を聴いたりしてリラックスしています。新しいことを学ぶのも好きですね。',
            '音楽はジャンルを問わず聞きますが、リラックスできるクラシックや、元気が出るポップスも好きです。',
            '野球道具にはこだわりがありますね。グローブやバットは自分の体の一部のような存在です。大切に手入れしています。',
            '休日は散歩をしたり、自然の中で過ごすことが多いです。心をリフレッシュできる時間を大切にしています。',
            'アクション映画やヒューマンドラマが好きですね。感動できる作品に出会うと、とても勉強になります。',
            'ゲームもたまにしますが、あまり長時間はやりません。適度にリフレッシュできる程度に楽しんでいます。',
            '簡単な料理なら作ります。特に和食を作ると、日本を思い出してほっとしますね。',
            '今はペットは飼っていませんが、将来的には考えているかもしれません。動物は癒されますね。',
            '読書もします。自己啓発本やスポーツ関連の本をよく読みます。新しい知識を得るのが楽しいです。',
            '車は好きですね。運転している時は集中できるし、良い気分転換になります。',
            
            # その他のカテゴリも同様に拡張...
        ]
        
        # 残りのデータを生成（合計100件以上になるように）
        while len(questions) < 100:
            # 既存の質問のバリエーションを作成
            base_questions = questions[:20]  # 最初の20個をベースに
            for q in base_questions:
                # 質問のバリエーションを作成
                variations = [
                    q.replace('？', 'について詳しく教えてください'),
                    q.replace('ますか', 'ますでしょうか'),
                    f"{q[:-1]}に関してはいかがですか？"
                ]
                for var in variations:
                    if len(questions) < 100 and var not in questions:
                        questions.append(var)
                        # 対応する回答も追加（既存回答のバリエーション）
                        base_answer = answers[len(answers) % len(base_questions)]
                        answers.append(f"{base_answer} より詳しくお話しすると、常に学び続ける姿勢を大切にしています。")
        
        return pd.DataFrame({
            'ID': range(1, len(questions) + 1),
            'Question': questions,
            'Answer': answers
        })
    
    def _extract_speech_patterns(self) -> Dict:
        """大幅拡張された話し方パターン"""
        return {
            'starters': [
                'そうですね', 'うーん', 'やっぱり', 'まあ', 'えーと',
                'そうですね...', 'んー', '実は', '正直に言うと',
                'いつも思うのは', '個人的には', 'これまでの経験から'
            ],
            'endings': [
                'と思います', 'かなと思います', 'じゃないかなと', 'ですね',
                'と感じています', 'と考えています', 'というのが正直なところです',
                'のかなと思います', 'と思っているんです', 'というふうに思います'
            ],
            'values': [
                '感謝', 'チーム', '成長', '挑戦', '継続', '努力', '学び',
                '仲間', '支え', '経験', '練習', '集中', '準備', '信頼'
            ],
            'humble': [
                'まだまだ', '勉強になります', 'ありがたい', 'おかげで',
                '至らないところも', '完璧ではありませんが', '日々勉強です',
                '皆さんのおかげで', 'まだ足りない部分も'
            ],
            'topics': {
                '野球': ['プレー', 'チーム', '練習', '試合', '技術', '戦術'],
                '食べ物': ['美味しい', '栄養', '健康', '日本の味', '新鮮'],
                '趣味': ['楽しい', 'リラックス', '気分転換', '新しい発見'],
                '人間関係': ['信頼', '支え', '感謝', '絆', 'コミュニケーション'],
                '将来': ['目標', '夢', '挑戦', '成長', '貢献']
            }
        }
    
    def _create_topic_templates(self) -> Dict:
        """トピック別の回答テンプレート"""
        return {
            '野球': [
                "{starter}、野球に関しては{value}を大切にしながら取り組んでいます{ending}。",
                "野球については、{humble}ですが、{value}することを心がけています{ending}。",
                "{starter}、これまでの経験から言うと、{value}が一番大切{ending}。"
            ],
            '食べ物': [
                "{starter}、食事に関しては体のことを考えて{value}を意識しています{ending}。",
                "食べ物については、{value}なものを選ぶようにしています{ending}。",
                "{starter}、{value}な食事を心がけることで、コンディション維持につながる{ending}。"
            ],
            '趣味': [
                "{starter}、{value}な時間を過ごすことで、良いリフレッシュになります{ending}。",
                "趣味については、{value}ことを大切にしています{ending}。",
                "{starter}、オフの時間は{value}を心がけています{ending}。"
            ],
            '人間関係': [
                "{starter}、{value}を基盤にした関係づくりを心がけています{ending}。",
                "人との関係では、{value}が一番大切{ending}。",
                "{starter}、{humble}ですが、{value}することを意識しています{ending}。"
            ],
            '将来': [
                "{starter}、将来に向けては{value}を持って取り組んでいきたい{ending}。",
                "これからについては、{value}し続けることが大切{ending}。",
                "{starter}、{value}という気持ちを忘れずに歩んでいきたい{ending}。"
            ]
        }
    
    def _detect_topic(self, query: str) -> str:
        """質問のトピックを判定"""
        keywords = {
            '野球': ['野球', '試合', 'バッティング', 'ピッチング', 'チーム', 'プレー', '練習', 'スランプ', 'ホームラン'],
            '食べ物': ['食べ物', '料理', '食事', '好き', 'うまい', '美味しい', 'グルメ', '栄養'],
            '趣味': ['趣味', '映画', '音楽', '読書', 'ゲーム', '休日', 'オフ', 'リラックス'],
            '人間関係': ['チームメイト', 'ファン', 'コーチ', '家族', '友人', '関係', '交流', 'コミュニケーション'],
            '将来': ['将来', '目標', '夢', '引退', '10年後', 'これから', '今後']
        }
        
        query_lower = query.lower()
        topic_scores = {}
        
        for topic, topic_keywords in keywords.items():
            score = sum(1 for keyword in topic_keywords if keyword in query_lower)
            if score > 0:
                topic_scores[topic] = score
        
        if topic_scores:
            return max(topic_scores.items(), key=lambda x: x[1])[0]
        return '一般'
    
    def _generate_improved_pattern_response(self, query: str) -> str:
        """改善されたパターン生成"""
        topic = self._detect_topic(query)
        
        if topic in self.topic_templates:
            template = random.choice(self.topic_templates[topic])
            starter = random.choice(self.ohtani_patterns['starters'])
            ending = random.choice(self.ohtani_patterns['endings'])
            
            # トピックに関連する価値観を選択
            if topic in self.ohtani_patterns['topics']:
                value = random.choice(self.ohtani_patterns['topics'][topic])
            else:
                value = random.choice(self.ohtani_patterns['values'])
            
            humble = random.choice(self.ohtani_patterns['humble'])
            
            return template.format(
                starter=starter,
                ending=ending,
                value=value,
                humble=humble
            )
        
        # 一般的な回答
        starter = random.choice(self.ohtani_patterns['starters'])
        ending = random.choice(self.ohtani_patterns['endings'])
        value = random.choice(self.ohtani_patterns['values'])
        
        general_templates = [
            f"{starter}、それについては{value}を大切にしながら向き合っています{ending}。",
            f"{query}に関しては、日々{value}することを心がけています{ending}。",
            f"{starter}、{value}という気持ちを持って取り組んでいます{ending}。"
        ]
        
        return random.choice(general_templates)
    
    def search(self, query: str, method: str = 'hybrid', threshold: float = 0.15, ai_provider: str = None, api_key: str = None) -> Dict:
        """改善されたRAG検索システム（Layer 5でのAI生成優先）"""
        
        search_results = []
        
        # Layer 1: TF-IDF検索（質問空間）
        if method in ['tfidf', 'hybrid']:
            tfidf_results = self.tfidf_search.search(query, top_k=3)
            if tfidf_results and tfidf_results[0][1] >= threshold:
                idx, score = tfidf_results[0]
                search_results = tfidf_results
                return {
                    'layer': 1,
                    'method': 'TF-IDF RAG',
                    'confidence': 'high' if score > 0.4 else 'medium',
                    'response': self.answers[idx],
                    'source': f"RAG検索 - ID {self.df.iloc[idx]['ID']}: {self.questions[idx][:50]}...",
                    'score': float(score),
                    'search_results': search_results,
                    'retrieved_docs': self._format_retrieved_docs(tfidf_results),
                    'needs_ai': False
                }
        
        # Layer 2: キーワード検索（質問空間）
        if method in ['keyword', 'hybrid']:
            keyword_results = self.keyword_search.search(query, top_k=3)
            if keyword_results and keyword_results[0][1] >= threshold * 0.5:
                idx, score = keyword_results[0]
                search_results = keyword_results
                return {
                    'layer': 2,
                    'method': 'キーワードRAG',
                    'confidence': 'medium',
                    'response': self.answers[idx],
                    'source': f"RAG検索 - ID {self.df.iloc[idx]['ID']}: {self.questions[idx][:50]}...",
                    'score': float(score),
                    'search_results': search_results,
                    'retrieved_docs': self._format_retrieved_docs(keyword_results),
                    'needs_ai': False
                }
        
        # Layer 3: 回答空間検索
        answer_results = self.answer_search.search(query, top_k=3)
        if answer_results and answer_results[0][1] >= threshold * 0.3:
            idx, score = answer_results[0]
            search_results = answer_results
            return {
                'layer': 3,
                'method': '回答空間RAG',
                'confidence': 'medium',
                'response': self.answers[idx],
                'source': f"RAG検索 - ID {self.df.iloc[idx]['ID']}: 回答から検索",
                'score': float(score),
                'search_results': search_results,
                'retrieved_docs': self._format_retrieved_docs(answer_results, answer_space=True),
                'needs_ai': False
            }
        
        # Layer 4: 複数文書を統合してRAG生成
        all_results = self.keyword_search.search(query, top_k=5)
        if all_results and all_results[0][1] >= 0.1:  # さらに低い閾値
            search_results = all_results
            aggregated_context = self._aggregate_multiple_docs(all_results[:3])
            return {
                'layer': 4,
                'method': '複数文書RAG',
                'confidence': 'medium',
                'response': aggregated_context,
                'source': f"RAG検索 - {len(all_results)}件の文書から統合生成",
                'score': float(all_results[0][1]) if all_results else 0.1,
                'search_results': search_results,
                'retrieved_docs': self._format_retrieved_docs(all_results),
                'needs_ai': False
            }
        
        # Layer 5: AI生成優先（RAG情報なしの新しい質問）
        # ここでAI APIが利用可能ならAI生成、そうでなければパターン生成
        if ai_provider and api_key:
            # AI生成用のコンテキストを準備（RAG情報なしver）
            ai_context = self.prepare_no_rag_ai_context(query)
            return {
                'layer': 5,
                'method': 'AI生成（新規質問）',
                'confidence': 'medium',
                'response': None,  # AI生成で後から設定
                'source': f'AI生成 - 新しい質問（トピック: {self._detect_topic(query)}）',
                'score': 0.2,  # AI生成なので少し高めのスコア
                'search_results': [],
                'retrieved_docs': [],
                'needs_ai': True,
                'ai_context': ai_context
            }
        else:
            # AI APIが利用できない場合のフォールバック
            generated_response = self._generate_improved_pattern_response(query)
            return {
                'layer': 5,
                'method': '改善パターン生成',
                'confidence': 'low',
                'response': generated_response,
                'source': f'パターン生成（トピック: {self._detect_topic(query)}）- APIキー未設定',
                'score': 0.1,
                'search_results': [],
                'retrieved_docs': [],
                'needs_ai': False
            }
    
    def _format_retrieved_docs(self, results: List[Tuple[int, float]], answer_space: bool = False) -> List[Dict]:
        """検索された文書の整形"""
        docs = []
        for idx, score in results:
            docs.append({
                'id': int(self.df.iloc[idx]['ID']),
                'question': self.questions[idx],
                'answer': self.answers[idx],
                'score': float(score),
                'search_type': '回答空間' if answer_space else '質問空間'
            })
        return docs
    
    def _aggregate_multiple_docs(self, results: List[Tuple[int, float]]) -> str:
        """複数文書からの情報統合（RAGの真価）- 改善版"""
        if not results:
            return self._generate_improved_pattern_response("一般的な質問")
        
        # 関連する複数の回答を取得
        relevant_answers = []
        for idx, score in results:
            if score > 0.05:  # より低い閾値で多くの文書を活用
                relevant_answers.append(self.answers[idx])
        
        if not relevant_answers:
            return self._generate_improved_pattern_response("一般的な質問")
        
        # 複数回答から共通要素を抽出して統合
        combined_keywords = []
        for answer in relevant_answers:
            keywords = re.findall(r'[\w\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+', answer)
            combined_keywords.extend(keywords)
        
        # 頻出キーワードを特定
        keyword_freq = Counter(combined_keywords)
        top_keywords = [k for k, v in keyword_freq.most_common(8) if v > 1 and len(k) > 1]
        
        # より自然な統合回答生成
        starter = random.choice(self.ohtani_patterns['starters'])
        ending = random.choice(self.ohtani_patterns['endings'])
        
        if top_keywords:
            # キーワードから適切な価値観を選択
            values_in_keywords = [k for k in top_keywords if k in self.ohtani_patterns['values']]
            if values_in_keywords:
                main_value = values_in_keywords[0]
            else:
                main_value = random.choice(self.ohtani_patterns['values'])
            
            # より具体的で自然な回答を生成
            templates = [
                f"{starter}、それについては{main_value}を大切にしながら、これまでの経験を活かして取り組んでいます。複数の場面で学んだことを総合すると、やはり継続することが重要{ending}。",
                f"{main_value}に関しては、いろいろな経験から学ばせていただきました。{starter}、今思うのは、一つ一つの積み重ねが大きな成果につながるということ{ending}。",
                f"{starter}、{main_value}という点では、チームメイトや周りの方々からも多くのことを教わりました。そういった経験を大切にしながら、これからも成長していきたい{ending}。"
            ]
        else:
            templates = [
                f"{starter}、その件については、日々の経験を通じて学んでいることが多いです。まだまだ勉強中ですが、前向きに取り組んでいきたい{ending}。",
                f"それについては、これまでいろいろな場面で考えさせられました。{starter}、自分なりに答えを見つけながら、成長していければと思っています{ending}。"
            ]
        
        return random.choice(templates)
    
    def prepare_ai_context(self, query: str, search_results: List[Tuple[int, float]]) -> str:
        """AI生成用コンテキスト準備 - 改善版（RAGありの場合）"""
        context_parts = []
        
        if search_results:
            context_parts.append("【参考となる大谷選手の過去の発言】")
            for i, (idx, score) in enumerate(search_results[:4], 1):  # より多くの参考資料
                context_parts.append(f"{i}. Q: {self.questions[idx]}")
                context_parts.append(f"   A: {self.answers[idx]}")
                context_parts.append(f"   類似度: {score:.3f}")
            context_parts.append("")
        
        # より詳細な話し方の特徴
        context_parts.extend([
            "【大谷翔平選手の話し方の詳細な特徴】",
            "- 謙虚で丁寧な口調（「そうですね」「と思います」「まだまだ」をよく使う）",
            "- チームメイトや周りの人への感謝を常に表現",
            "- 成長・学び・継続・努力を大切にする姿勢",
            "- 前向きで誠実、時に照れるような素直な答え方",
            "- 野球での具体的な経験を交えながら答える",
            "- 困難な質問にも真摯に向き合う姿勢",
            "- 未来に向けての建設的な考え方",
            "",
            f"質問のトピック: {self._detect_topic(query)}",
            f"質問: {query}",
            "",
            "【指示】",
            "あなたは大谷翔平選手として、上記の参考発言と話し方の特徴を活かして、",
            "自然で温かみのある回答を200-300文字で作成してください。",
            "参考資料の内容を踏まえつつ、質問に対して大谷選手らしい誠実で前向きな回答をしてください：",
        ])
        
        return "\n".join(context_parts)

    def prepare_no_rag_ai_context(self, query: str) -> str:
        """Layer 5用：RAG情報なしでのAI生成コンテキスト（80文字回答用）"""
        topic = self._detect_topic(query)
        
        context_parts = [
            "【大谷翔平選手として回答してください】",
            "",
            "【話し方の特徴】",
            "- 謙虚で丁寧（「そうですね」「と思います」）",
            "- 感謝の気持ちを表現",
            "- 前向きで誠実",
            "- 成長・努力・チームワークを重視",
            "",
            f"質問のトピック: {topic}",
            f"質問: {query}",
            "",
            "【重要な指示】",
            "- 大谷翔平選手として自然に回答",
            "- 日本語で70-90文字程度の簡潔な回答",
            "- 謙虚さと前向きさを含めて",
            "- 具体的すぎる情報は避けて一般的な姿勢で答える",
            "",
            "回答："
        ]
        
        return "\n".join(context_parts)
        """AI生成用コンテキスト準備 - 改善版"""
        context_parts = []
        
        if search_results:
            context_parts.append("【参考となる大谷選手の過去の発言】")
            for i, (idx, score) in enumerate(search_results[:4], 1):  # より多くの参考資料
                context_parts.append(f"{i}. Q: {self.questions[idx]}")
                context_parts.append(f"   A: {self.answers[idx]}")
                context_parts.append(f"   類似度: {score:.3f}")
            context_parts.append("")
        
        # より詳細な話し方の特徴
        context_parts.extend([
            "【大谷翔平選手の話し方の詳細な特徴】",
            "- 謙虚で丁寧な口調（「そうですね」「と思います」「まだまだ」をよく使う）",
            "- チームメイトや周りの人への感謝を常に表現",
            "- 成長・学び・継続・努力を大切にする姿勢",
            "- 前向きで誠実、時に照れるような素直な答え方",
            "- 野球での具体的な経験を交えながら答える",
            "- 困難な質問にも真摯に向き合う姿勢",
            "- 未来に向けての建設的な考え方",
            "",
            f"質問のトピック: {self._detect_topic(query)}",
            f"質問: {query}",
            "",
            "【指示】",
            "あなたは大谷翔平選手として、上記の参考発言と話し方の特徴を活かして、",
            "自然で温かみのある回答を200-300文字で作成してください。",
            "参考資料の内容を踏まえつつ、質問に対して大谷選手らしい誠実で前向きな回答をしてください：",
        ])
        
        return "\n".join(context_parts)

# AI API呼び出し関数（Layer 5専用バージョン追加）
def call_gemini_api_layer5(prompt: str, api_key: str) -> Optional[str]:
    """Gemini API呼び出し - Layer 5専用（80文字回答）"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=150,  # 短い回答用
                temperature=0.9,        # より創造性を高める
                top_p=0.95,
                top_k=50
            )
        )
        
        return response.text if hasattr(response, 'text') else None
    except Exception as e:
        return f"Gemini APIエラー: {str(e)}"

def call_openai_api_layer5(prompt: str, api_key: str) -> Optional[str]:
    """OpenAI API呼び出し - Layer 5専用（80文字回答）"""
    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'gpt-3.5-turbo',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 120,      # 短い回答用
            'temperature': 0.9,     # より創造性を高める
            'top_p': 0.95,
            'frequency_penalty': 0.4,  # 繰り返しを更に減らす
            'presence_penalty': 0.3
        }
        
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"OpenAI APIエラー: {response.status_code}"
    except Exception as e:
        return f"OpenAI API接続エラー: {str(e)}"

# AI API呼び出し関数（改善版）
def call_gemini_api(prompt: str, api_key: str) -> Optional[str]:
    """Gemini API呼び出し - 改善版"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=400,  # より長い回答を許可
                temperature=0.8,        # より創造性を高める
                top_p=0.9,
                top_k=40
            )
        )
        
        return response.text if hasattr(response, 'text') else None
    except Exception as e:
        return f"Gemini APIエラー: {str(e)}"

def call_openai_api(prompt: str, api_key: str) -> Optional[str]:
    """OpenAI API呼び出し - 改善版"""
    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'gpt-3.5-turbo',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 350,      # より長い回答を許可
            'temperature': 0.8,     # より創造性を高める
            'top_p': 0.9,
            'frequency_penalty': 0.3,  # 繰り返しを減らす
            'presence_penalty': 0.2
        }
        
        response = requests.post(
            'https://api.openai.com/v1/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return f"OpenAI APIエラー: {response.status_code}"
    except Exception as e:
        return f"OpenAI API接続エラー: {str(e)}"

# メイン関数
def main():
    st.title("AI大谷 - 改善版")
    st.subheader("🚀 多様性向上RAG + 生成AI ハイブリッドシステム")
    
    # 改善点の説明
    with st.expander("🔧 この版の改善点"):
        st.markdown("""
        **主な改善点：**
        1. **サンプルデータを20件→100件以上に拡張**
        2. **検索閾値を0.3→0.15に下げて、より多くのRAG検索を成功**
        3. **トピック別回答テンプレートを追加（野球、食べ物、趣味など）**
        4. **話し方パターンを大幅拡張（謙虚表現、価値観など）**
        5. **AI API設定を最適化（temperature上昇、頻度ペナルティ追加）**
        6. **複数文書統合ロジックの改善**
        """)
    
    # サイドバー設定
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # 検索設定
        search_method = st.selectbox(
            "検索方法",
            options=['hybrid', 'tfidf', 'keyword'],
            index=0,
            help="hybrid: 複数手法組み合わせ（推奨）"
        )
        
        # 改善された閾値設定
        threshold = st.slider(
            "検索閾値", 
            0.05, 0.5, 0.15, 0.02,
            help="低い値ほどより多くの文書がマッチ（推奨: 0.15）"
        )
        
        st.divider()
        
        # AI API設定
        st.subheader("🤖 生成AI設定")
        ai_provider = st.selectbox("AIプロバイダー", ["なし", "Gemini", "OpenAI"])
        
        api_key = ""
        if ai_provider == "Gemini":
            api_key = st.text_input(
                "Gemini API Key",
                type="password",
                value=os.getenv("GEMINI_API_KEY", ""),
                help="Gemini APIキーを入力してください"
            )
        elif ai_provider == "OpenAI":
            api_key = st.text_input(
                "OpenAI API Key", 
                type="password",
                value=os.getenv("OPENAI_API_KEY", ""),
                help="OpenAI APIキーを入力してください"
            )
        
        use_ai = ai_provider != "なし" and bool(api_key)
        
        if use_ai:
            st.success(f"✅ {ai_provider} API 有効")
        else:
            st.info("💡 APIキー未設定: 改善されたパターン生成を使用")
    
    # RAGシステム初期化
    @st.cache_resource
    def load_rag_system():
        return ImprovedOhtaniRAG('ohtani_rag_final.csv')
    
    with st.spinner("🚀 改善システム初期化中..."):
        rag = load_rag_system()
    
    st.success(f"✅ 初期化完了！ ({len(rag.df)}件のデータを読み込み)")
    
    # メイン画面
    st.markdown("---")
    
    # 質問入力
    query = st.text_input(
        "💬 大谷選手に質問してください:",
        placeholder="例: 今日のトレーニングはどうでしたか？",
        help="どんな質問でも大谷選手風に回答します（回答の多様性が向上）"
    )
    
    # 操作ボタン
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
    
    with col1:
        search_btn = st.button("🔍 質問する", type="primary")
    with col2:
        random_btn = st.button("🎲 ランダム")
    with col3:
        if st.button("🔄 リセット"):
            st.rerun()
    with col4:
        show_stats = st.button("📊 統計")
    
    # ランダム質問（拡張版）
    if random_btn:
        sample_queries = [
            "今日の調子はいかがですか？",
            "最近読んだ本について教えてください",
            "チームの雰囲気はどうですか？",
            "好きなアニメはありますか？",
            "日本の家族と連絡は取っていますか？",
            "アメリカの生活で驚いたことは？",
            "子供の頃の夢は何でしたか？",
            "尊敬している人について教えてください",
            "ストレス発散方法は？",
            "将来日本でプレーしたいですか？",
            "好きな季節はいつですか？",
            "料理で得意なものはありますか？"
        ]
        query = random.choice(sample_queries)
        search_btn = True
    
    # 検索実行
    if search_btn and query.strip():
        with st.spinner("🤖 検索・生成中..."):
            start_time = time.time()
            
            # RAG検索
            result = rag.search(query, method=search_method, threshold=threshold, ai_provider=ai_provider, api_key=api_key)
            search_time = time.time() - start_time
            
            # AI生成処理の分岐
            ai_response = None
            ai_time = 0
            
            if result.get('needs_ai') and use_ai:
                # Layer 5: 新規質問のAI生成
                ai_start = time.time()
                if ai_provider == "Gemini":
                    ai_response = call_gemini_api_layer5(result['ai_context'], api_key)
                elif ai_provider == "OpenAI":
                    ai_response = call_openai_api_layer5(result['ai_context'], api_key)
                
                ai_time = time.time() - ai_start
                
                if ai_response and not ai_response.startswith("API"):
                    result['response'] = ai_response.strip()
                    result['method'] = f"{ai_provider} AI生成（新規質問）"
                    result['confidence'] = 'medium'
                    st.success(f"✅ Layer 5 AI生成成功: 新しい質問に対してAIが回答生成 ({ai_time:.2f}秒)")
                else:
                    # AI生成失敗時のフォールバック
                    result['response'] = rag._generate_improved_pattern_response(query)
                    result['method'] = '改善パターン生成（AI失敗）'
                    st.warning("⚠️ AI生成失敗、パターン生成にフォールバック")
                    
            elif result.get('search_results') and use_ai and result['layer'] <= 4:
                # Layer 1-4: RAG情報ありのAI強化
                ai_start = time.time()
                context = rag.prepare_ai_context(query, result['search_results'])
                
                if ai_provider == "Gemini":
                    ai_response = call_gemini_api(context, api_key)
                elif ai_provider == "OpenAI":
                    ai_response = call_openai_api(context, api_key)
                
                ai_time = time.time() - ai_start
                
                if ai_response and not ai_response.startswith("API"):
                    st.info(f"✅ RAG+AI強化: {len(result.get('retrieved_docs', []))}件の文書から生成 ({ai_time:.2f}秒)")
            
            # 結果表示
            st.markdown("---")
            
            # パフォーマンス情報
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                layer_colors = {1: '🟢', 2: '🟡', 3: '🟠', 4: '🔵', 5: '🟣'}
                st.metric("レイヤー", f"{layer_colors.get(result['layer'], '⚪')} Layer {result['layer']}")
            with col2:
                confidence_colors = {'high': '🟢', 'medium': '🟡', 'low': '🔵'}
                st.metric("信頼度", f"{confidence_colors[result['confidence']]} {result['confidence']}")
            with col3:
                st.metric("スコア", f"{result['score']:.3f}")
            with col4:
                if ai_time > 0:
                    st.metric("AI生成時間", f"{ai_time:.2f}秒")
                else:
                    st.metric("検索時間", f"{search_time:.2f}秒")
            
            # 回答表示の改善
            if result['layer'] == 5 and result.get('needs_ai') and result['response']:
                # Layer 5でのAI生成成功
                st.markdown("### 🤖 AI大谷（新規質問）")
                st.markdown(f"> {result['response']}")
                
                st.success(f"🎯 新しい質問に対してAIが大谷選手風に回答生成")
                
            elif ai_response and not ai_response.startswith("API") and result['layer'] <= 4:
                # Layer 1-4でのRAG+AI強化
                st.markdown("### 🤖 RAG + AI生成回答")
                st.markdown(f"> {ai_response}")
                
                st.success(f"🔍 RAG検索成功: {len(result.get('retrieved_docs', []))}件の関連文書を活用")
                
                with st.expander("🔍 RAG検索詳細"):
                    st.markdown(f"**検索方法:** {result['method']}")
                    st.markdown(f"**元の回答:** {result['response']}")
                    st.markdown(f"**出典:** {result['source']}")
                    
                    if result.get('retrieved_docs'):
                        st.markdown("**検索された関連文書:**")
                        for i, doc in enumerate(result['retrieved_docs'][:4], 1):
                            st.markdown(f"{i}. スコア: {doc['score']:.3f} | ID: {doc['id']}")
                            st.markdown(f"   Q: {doc['question']}")
                            st.markdown(f"   A: {doc['answer'][:100]}...")
            else:
                # 通常のRAG回答またはパターン生成
                st.markdown("### 💬 AI大谷")
                st.markdown(f"> {result['response']}")
                
                if result['layer'] <= 4:
                    st.info(f"🔍 RAG検索: {result['method']}で関連文書を発見")
                elif result['layer'] == 5:
                    if use_ai:
                        st.warning("⚠️ 新規質問でしたが、AI生成に失敗しました")
                    else:
                        st.info("💡 新しい質問です。より自然な回答にはAPIキーを設定してください")
                
                if ai_response and ai_response.startswith("API"):
                    st.error(f"🚫 AI生成失敗: {ai_response}")
            
            # レイヤー別の説明（更新）
            layer_explanations = {
                1: "🟢 TF-IDFによる高精度マッチング",
                2: "🟡 キーワードによる中精度マッチング", 
                3: "🟠 回答空間からの関連検索",
                4: "🔵 複数文書統合による生成",
                5: "🟣 AI生成（新規質問）" if use_ai else "🟣 パターン生成（新規質問）"
            }
            
            st.info(f"使用したレイヤー: {layer_explanations.get(result['layer'], 'その他')}")
            
            # Layer 5の特別説明
            if result['layer'] == 5:
                if use_ai and result.get('needs_ai'):
                    st.info("🚀 **Layer 5**: RAG検索で関連文書が見つからなかった新しい質問に対して、AIが大谷選手風の回答を生成しました！")
                elif use_ai:
                    st.info("🤖 **Layer 5**: AI生成が利用可能でしたが、パターン生成で十分な回答ができました")
                else:
                    st.info("💡 **Layer 5**: 新しい質問です。AIキーを設定すると、より自然で多様な回答が可能になります")
            
            # 詳細情報
            with st.expander("📝 詳細情報"):
                detailed_info = {
                    "検索レイヤー": result['layer'],
                    "検索方法": result['method'], 
                    "信頼度": result['confidence'],
                    "スコア": result['score'],
                    "出典": result['source'],
                    "検索時間": f"{search_time:.3f}秒",
                    "AI生成時間": f"{ai_time:.3f}秒" if ai_time > 0 else "未使用",
                    "検索された文書数": len(result.get('retrieved_docs', [])),
                    "検出されたトピック": rag._detect_topic(query),
                    "AI生成が必要": result.get('needs_ai', False),
                    "回答文字数": len(result['response']) if result['response'] else 0
                }
                st.json(detailed_info)
    
    # 統計情報表示
    if show_stats:
        st.markdown("---")
        st.markdown("### 📊 システム統計")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("総データ数", len(rag.df))
            st.metric("語彙サイズ", len(rag.tfidf_search.vocab))
            st.metric("トピックテンプレート数", len(rag.topic_templates))
        with col2:
            st.metric("キーワード数", len(rag.keyword_search.keyword_index))
            st.metric("話し方パターン数", len(rag.ohtani_patterns['starters']) + len(rag.ohtani_patterns['endings']))
            st.metric("バージョン", "Layer5-AI強化版 v3.0")
    
    # 改善されたサンプル質問セクション
    st.markdown("---")
    st.markdown("### 💡 カテゴリ別サンプル質問")
    
    improved_sample_categories = {
        "⚾ 野球・競技": [
            "今シーズンの手応えはいかがですか？",
            "バッティングフォームで最近変えたことは？",
            "チームメイトとの連携で意識していることは？",
            "プレッシャーのかかる場面での心構えは？"
        ],
        "🍱 食事・健康": [
            "体調管理で気をつけていることは？", 
            "アメリカで好きになった食べ物は？",
            "日本食が恋しくなることはありますか？",
            "栄養面で意識していることは？"
        ],
        "🎯 プライベート・趣味": [
            "最近ハマっていることはありますか？",
            "休日のリラックス方法は？",
            "好きな音楽や映画はありますか？",
            "新しく挑戦してみたいことは？"
        ],
        "👥 人間関係・コミュニケーション": [
            "ファンとの交流で印象的だったことは？",
            "コーチとのコミュニケーションで大切にしていることは？",
            "日本の家族や友人とは連絡を取り合っていますか？",
            "言葉の壁を感じることはありますか？"
        ],
        "🌟 将来・夢・価値観": [
            "10年後の自分はどうなっていたいですか？",
            "野球を通じて伝えたいことは？",
            "人生で一番大切にしている価値観は？",
            "次世代の選手たちにアドバイスはありますか？"
        ]
    }
    
    for category, questions in improved_sample_categories.items():
        with st.expander(category):
            for i, q in enumerate(questions):
                if st.button(q, key=f"{category}_{i}"):
                    result = rag.search(q, method=search_method, threshold=threshold)
                    st.write(f"**質問:** {q}")
                    st.write(f"**回答:** {result['response']}")
                    st.write(f"**レイヤー:** {result['layer']} | **スコア:** {result['score']:.3f}")

if __name__ == "__main__":
    main()