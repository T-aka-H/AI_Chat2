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
    page_title="AI大谷 - 高速版",
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
class FastOhtaniRAG:
    """高速大谷翔平RAGシステム"""
    
    def __init__(self, csv_path: str):
        self.df = self._load_data(csv_path)
        self.questions = self.df['Question'].fillna('').astype(str).tolist()
        self.answers = self.df['Answer'].fillna('').astype(str).tolist()
        
        # 検索システム初期化
        self.tfidf_search = LightweightTextSearch(self.questions)
        self.keyword_search = KeywordSearch(self.questions)
        self.answer_search = KeywordSearch(self.answers)
        
        # 大谷選手の話し方パターン
        self.ohtani_patterns = self._extract_speech_patterns()
    
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
            return self._create_sample_data()
    
    def _create_sample_data(self) -> pd.DataFrame:
        """サンプルデータ"""
        return pd.DataFrame({
            'ID': range(1, 21),
            'Question': [
                '野球以外で興味のあることはありますか？',
                'オフシーズンはどう過ごしていますか？',
                '好きな食べ物は何ですか？',
                '将来の目標について教えてください',
                'チームメイトとの関係はいかがですか？',
                '困難な時期をどう乗り越えますか？',
                '日本とアメリカの違いは？',
                'ファンへのメッセージをお願いします',
                'トレーニングで大切にしていることは？',
                '野球を始めたきっかけは？',
                'リラックス方法は？',
                '尊敬する選手はいますか？',
                '子供たちへのアドバイスは？',
                'プレッシャーを感じることは？',
                '今シーズンの目標は？',
                'コーチとの関係について',
                'けがをした時の気持ちは？',
                'オールスターゲームの感想は？',
                '野球の魅力とは？',
                'これからの野球界について'
            ],
            'Answer': [
                'そうですね、料理をするのが好きですね。新しいレシピに挑戦することで、野球以外でも成長できると思っています。',
                'トレーニングはもちろんですが、リラックスすることも大切にしています。読書をしたり、映画を見たりしています。',
                '和食が一番好きですね。特に母が作ってくれた料理の味は忘れられません。',
                '常に成長し続けることが目標です。野球を通じて多くの人に影響を与えられる選手になりたいです。',
                'チームメイトはみんな素晴らしい人たちです。お互いを高め合える関係を築けていると思います。',
                '困難な時こそ、基本に立ち戻ることを大切にしています。そして、支えてくれる人たちへの感謝を忘れずに。',
                '文化の違いはありますが、野球への情熱は同じです。どちらの国からも学ぶことがたくさんあります。',
                'いつも応援してくださって、本当にありがとうございます。皆さんの声援が力になっています。',
                '継続することが一番大切だと思います。小さなことの積み重ねが大きな成果につながります。',
                '父の影響が大きかったです。野球の楽しさを教えてもらいました。',
                '自然の中で過ごすことが多いですね。散歩をしたり、空を眺めたりしています。',
                'イチロー選手には本当に多くのことを学ばせていただきました。',
                '好きなことを見つけて、それを大切にしてほしいです。そして諦めずに続けてください。',
                'プレッシャーは感じますが、それを楽しめるようになりました。',
                'チーム一丸となって、良い結果を残したいと思います。',
                'コーチからはたくさんのアドバイスをもらっています。とても感謝しています。',
                'けがは辛いですが、それも経験の一つだと考えています。',
                'ファンの皆さんと一緒に楽しい時間を過ごせました。',
                '野球は人と人をつなげる素晴らしいスポーツだと思います。',
                '若い選手たちの成長が楽しみです。野球界全体がより良くなることを願っています。'
            ]
        })
    
    def _extract_speech_patterns(self) -> Dict:
        """大谷選手の話し方パターン抽出"""
        return {
            'starters': ['そうですね', 'うーん', 'やっぱり', 'まあ'],
            'endings': ['と思います', 'かなと思います', 'じゃないかなと', 'ですね'],
            'values': ['感謝', 'チーム', '成長', '挑戦', '継続', '努力'],
            'humble': ['まだまだ', '勉強になります', 'ありがたい', 'おかげで']
        }
    
    def search(self, query: str, method: str = 'hybrid', threshold: float = 0.3) -> Dict:
        """RAG検索システム - Retrieval-Augmented Generation"""
        
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
                    'confidence': 'high' if score > 0.5 else 'medium',
                    'response': self.answers[idx],
                    'source': f"RAG検索 - ID {self.df.iloc[idx]['ID']}: {self.questions[idx][:50]}...",
                    'score': float(score),
                    'search_results': search_results,
                    'retrieved_docs': self._format_retrieved_docs(tfidf_results)
                }
        
        # Layer 2: キーワード検索（質問空間）
        if method in ['keyword', 'hybrid']:
            keyword_results = self.keyword_search.search(query, top_k=3)
            if keyword_results and keyword_results[0][1] >= threshold * 0.7:
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
                    'retrieved_docs': self._format_retrieved_docs(keyword_results)
                }
        
        # Layer 3: 回答空間検索
        answer_results = self.answer_search.search(query, top_k=3)
        if answer_results and answer_results[0][1] >= threshold * 0.5:
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
                'retrieved_docs': self._format_retrieved_docs(answer_results, answer_space=True)
            }
        
        # Layer 4: 複数文書を統合してRAG生成
        all_results = self.keyword_search.search(query, top_k=5)
        if all_results:
            search_results = all_results
            # 複数の関連文書を取得して統合
            aggregated_context = self._aggregate_multiple_docs(all_results[:3])
            return {
                'layer': 4,
                'method': '複数文書RAG',
                'confidence': 'medium',
                'response': aggregated_context,
                'source': f"RAG検索 - {len(all_results)}件の文書から統合生成",
                'score': float(all_results[0][1]) if all_results else 0.1,
                'search_results': search_results,
                'retrieved_docs': self._format_retrieved_docs(all_results)
            }
        
        # Layer 5: パターン生成（RAG失敗時のフォールバック）
        generated_response = self._generate_pattern_response(query)
        return {
            'layer': 5,
            'method': 'パターン生成（非RAG）',
            'confidence': 'low',
            'response': generated_response,
            'source': '大谷選手の発言パターンから生成（RAG情報なし）',
            'score': 0.1,
            'search_results': [],
            'retrieved_docs': []
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
        """複数文書からの情報統合（RAGの真価）"""
        if not results:
            return self._generate_pattern_response("一般的な質問")
        
        # 関連する複数の回答を取得
        relevant_answers = []
        for idx, score in results:
            if score > 0.1:  # 最低限の関連性
                relevant_answers.append(self.answers[idx])
        
        if not relevant_answers:
            return self._generate_pattern_response("一般的な質問")
        
        # 複数回答から共通要素を抽出して統合
        combined_keywords = []
        for answer in relevant_answers:
            keywords = re.findall(r'[\w\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+', answer)
            combined_keywords.extend(keywords)
        
        # 頻出キーワードを特定
        keyword_freq = Counter(combined_keywords)
        top_keywords = [k for k, v in keyword_freq.most_common(5) if v > 1]
        
        # 統合回答生成
        starter = random.choice(self.ohtani_patterns['starters'])
        value = random.choice(top_keywords) if top_keywords else random.choice(self.ohtani_patterns['values'])
        ending = random.choice(self.ohtani_patterns['endings'])
        
        return f"{starter}、それについては{value}を大切にしながら取り組んでいます。複数の経験から学んだことを活かして、これからも成長していきたい{ending}。"
    
    def _generate_pattern_response(self, query: str) -> str:
        """パターン生成"""
        starter = random.choice(self.ohtani_patterns['starters'])
        ending = random.choice(self.ohtani_patterns['endings'])
        value = random.choice(self.ohtani_patterns['values'])
        
        templates = [
            f"{starter}、{query}については、{value}を大切に{ending}。",
            f"{query}に関しては、まだまだ学ぶことが多い{ending}。",
            f"{starter}、{query}というのは、とても大切なこと{ending}。"
        ]
        
        return random.choice(templates)
    
    def prepare_ai_context(self, query: str, search_results: List[Tuple[int, float]]) -> str:
        """AI生成用コンテキスト準備"""
        context_parts = []
        
        if search_results:
            context_parts.append("【参考となる大谷選手の過去の発言】")
            for i, (idx, score) in enumerate(search_results[:3], 1):
                context_parts.append(f"{i}. Q: {self.questions[idx]}")
                context_parts.append(f"   A: {self.answers[idx]}")
            context_parts.append("")
        
        context_parts.extend([
            "【大谷翔平選手の話し方の特徴】",
            "- 謙虚で丁寧な口調（「そうですね」「と思います」をよく使う）",
            "- チームメイトや周りの人への感謝を忘れない",
            "- 成長や学び、継続を大切にする姿勢",
            "- 前向きで誠実な答え方",
            "- 野球での経験を交えながら答える",
            "",
            f"質問: {query}",
            "",
            "あなたは大谷翔平選手として、上記の特徴を活かして150-250文字で自然に回答してください：",
        ])
        
        return "\n".join(context_parts)

# AI API呼び出し関数
def call_gemini_api(prompt: str, api_key: str) -> Optional[str]:
    """Gemini API呼び出し"""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                max_output_tokens=300,
                temperature=0.7
            )
        )
        
        return response.text if hasattr(response, 'text') else None
    except Exception as e:
        return f"Gemini APIエラー: {str(e)}"

def call_openai_api(prompt: str, api_key: str) -> Optional[str]:
    """OpenAI API呼び出し"""
    try:
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        data = {
            'model': 'gpt-3.5-turbo',
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': 250,
            'temperature': 0.7
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
    st.title("AI大谷")
    st.subheader("🚀 高速RAG + 生成AI ハイブリッドシステム")
    
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
        
        threshold = st.slider("検索閾値", 0.1, 0.8, 0.3, 0.05)
        
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
            st.info("💡 APIキー未設定: パターン生成を使用")
    
    # RAGシステム初期化
    @st.cache_resource
    def load_rag_system():
        return FastOhtaniRAG('ohtani_rag_final.csv')
    
    with st.spinner("🚀 システム初期化中..."):
        rag = load_rag_system()
    
    st.success(f"✅ 初期化完了！ ({len(rag.df)}件のデータを読み込み)")
    
    # メイン画面
    st.markdown("---")
    
    # 質問入力
    query = st.text_input(
        "💬 大谷選手に質問してください:",
        placeholder="例: 野球以外で興味のあることはありますか？",
        help="どんな質問でも大谷選手風に回答します"
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
    
    # ランダム質問
    if random_btn:
        sample_queries = [
            "野球以外の趣味について教えてください",
            "オフシーズンの過ごし方は？",
            "好きな食べ物はありますか？",
            "将来の目標を聞かせてください",
            "ファンの皆さんへメッセージを",
            "困難を乗り越える秘訣は？",
            "チームメイトとの関係について",
            "トレーニングで心がけていることは？"
        ]
        query = random.choice(sample_queries)
        search_btn = True
    
    # 検索実行
    if search_btn and query.strip():
        with st.spinner("🤖 検索・生成中..."):
            start_time = time.time()
            
            # RAG検索
            result = rag.search(query, method=search_method, threshold=threshold)
            search_time = time.time() - start_time
            
            # AI生成（設定されている場合）- これが真のRAG！
            ai_response = None
            if use_ai and result.get('search_results'):
                ai_start = time.time()
                # RAG: 検索結果を使ってコンテキスト強化
                context = rag.prepare_ai_context(query, result['search_results'])
                
                if ai_provider == "Gemini":
                    ai_response = call_gemini_api(context, api_key)
                elif ai_provider == "OpenAI":
                    ai_response = call_openai_api(context, api_key)
                
                ai_time = time.time() - ai_start
                
                # RAG成功の表示
                if ai_response and not ai_response.startswith("API"):
                    st.info(f"✅ RAG成功: {len(result.get('retrieved_docs', []))}件の文書から生成 ({ai_time:.2f}秒)")
            
            # 結果表示
            st.markdown("---")
            
            # パフォーマンス情報
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("レイヤー", f"Layer {result['layer']}")
            with col2:
                confidence_colors = {'high': '🟢', 'medium': '🟡', 'low': '🔵'}
                st.metric("信頼度", f"{confidence_colors[result['confidence']]} {result['confidence']}")
            with col3:
                st.metric("スコア", f"{result['score']:.3f}")
            with col4:
                st.metric("検索時間", f"{search_time:.2f}秒")
            
            # 回答表示
            if ai_response and not ai_response.startswith("API"):
                st.markdown("### 🤖 RAG + AI生成回答")
                st.markdown(f"> {ai_response}")
                
                st.success(f"🔍 RAG検索成功: {len(result.get('retrieved_docs', []))}件の関連文書を発見")
                
                with st.expander("🔍 RAG検索詳細"):
                    st.markdown(f"**検索方法:** {result['method']}")
                    st.markdown(f"**元の回答:** {result['response']}")
                    st.markdown(f"**出典:** {result['source']}")
                    
                    # 検索された文書一覧
                    if result.get('retrieved_docs'):
                        st.markdown("**検索された関連文書:**")
                        for i, doc in enumerate(result['retrieved_docs'][:3], 1):
                            st.markdown(f"{i}. スコア: {doc['score']:.3f}")
                            st.markdown(f"   Q: {doc['question']}")
                            st.markdown(f"   A: {doc['answer'][:100]}...")
            else:
                st.markdown("### 💬 RAG検索回答")
                st.markdown(f"> {result['response']}")
                
                if result['layer'] <= 4:
                    st.info(f"🔍 RAG検索: {result['method']}で関連文書を発見")
                else:
                    st.warning("⚠️ RAG検索で関連文書が見つからず、パターン生成を使用")
                
                if ai_response and ai_response.startswith("API"):
                    st.error(f"🚫 AI生成失敗: {ai_response}")
                elif not use_ai:
                    st.info("💡 より高品質な回答には、サイドバーでAI APIキーを設定してください")
            
            # 詳細情報
            with st.expander("📝 詳細情報"):
                st.json({
                    "検索レイヤー": result['layer'],
                    "検索方法": result['method'], 
                    "信頼度": result['confidence'],
                    "スコア": result['score'],
                    "出典": result['source'],
                    "検索時間": f"{search_time:.3f}秒"
                })
    
    # 統計情報表示
    if show_stats:
        st.markdown("---")
        st.markdown("### 📊 システム統計")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("総データ数", len(rag.df))
            st.metric("語彙サイズ", len(rag.tfidf_search.vocab))
        with col2:
            st.metric("キーワード数", len(rag.keyword_search.keyword_index))
            st.metric("メモリ効率", "軽量版")
    
    # サンプル質問セクション
    st.markdown("---")
    st.markdown("### 💡 サンプル質問")
    
    sample_categories = {
        "🏃‍♂️ 野球・スポーツ": [
            "今シーズンの目標は？",
            "トレーニングで大切にしていることは？",
            "プレッシャーとどう向き合っていますか？"
        ],
        "🎯 プライベート": [
            "オフの日はどう過ごしますか？", 
            "好きな食べ物は？",
            "リラックス方法は？"
        ],
        "🌟 人生観": [
            "将来の夢について教えてください",
            "困難を乗り越える秘訣は？",
            "大切にしている価値観は？"
        ]
    }
    
    for category, questions in sample_categories.items():
        with st.expander(category):
            for i, q in enumerate(questions):
                if st.button(q, key=f"{category}_{i}"):
                    # 質問を実行
                    result = rag.search(q, method=search_method, threshold=threshold)
                    st.write(f"**質問:** {q}")
                    st.write(f"**回答:** {result['response']}")

if __name__ == "__main__":
    main()