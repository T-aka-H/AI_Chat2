import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from typing import List, Tuple, Dict
import os

# 追加: 埋め込みと日本語トークナイズ
try:
    from sentence_transformers import SentenceTransformer
    _EMBED_AVAILABLE = True
except Exception:
    _EMBED_AVAILABLE = False

try:
    from janome.tokenizer import Tokenizer as JanomeTokenizer
    _JANOME_AVAILABLE = True
except Exception:
    _JANOME_AVAILABLE = False

# 設定
st.set_page_config(
    page_title="大谷翔平 AI Chat - 3層段階RAGシステム",
    page_icon="⚾",
    layout="wide"
)

# 日本語トークナイズ（Janomeがあれば利用）
def tokenize_ja(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    if _JANOME_AVAILABLE:
        t = JanomeTokenizer()
        return [token.surface for token in t.tokenize(text)]
    # フォールバック: 簡易分割
    return re.findall(r"\w+|[\u3040-\u30ff\u4e00-\u9fff]+", text)

# MMR (Maximal Marginal Relevance) 実装
def mmr(query_vec: np.ndarray, doc_vecs: np.ndarray, top_k: int = 5, lambda_mult: float = 0.5) -> List[int]:
    selected = []
    sim_to_query = cosine_similarity(query_vec.reshape(1, -1), doc_vecs)[0]
    candidates = list(range(len(doc_vecs)))
    if len(candidates) == 0:
        return []

    # 1つ目は最も近いもの
    best_idx = int(np.argmax(sim_to_query))
    selected.append(best_idx)
    candidates.remove(best_idx)

    while len(selected) < min(top_k, len(doc_vecs)) and candidates:
        mmr_scores = []
        for c in candidates:
            redundancy = max([cosine_similarity(doc_vecs[c].reshape(1, -1), doc_vecs[s].reshape(1, -1))[0][0] for s in selected]) if selected else 0.0
            score = lambda_mult * sim_to_query[c] - (1 - lambda_mult) * redundancy
            mmr_scores.append((c, score))
        mmr_scores.sort(key=lambda x: x[1], reverse=True)
        best = mmr_scores[0][0]
        selected.append(best)
        candidates.remove(best)
    return selected

class OhtaniRAGSystem:
    def __init__(self, csv_path: str, use_embeddings: bool, embed_model_name: str):
        """大谷翔平RAGシステムの初期化"""
        # CSV読み込み（RAG配下が最小行なら親ディレクトリをフォールバック）
        df = pd.read_csv(csv_path)
        if len(df) < 50:
            parent_path = os.path.join(os.path.dirname(os.path.dirname(csv_path)), "ohtani_rag_final.csv")
            if os.path.exists(parent_path):
                df = pd.read_csv(parent_path)
        self.df = df

        # TF-IDF（日本語トークナイザ対応）
        analyzer = None
        if _JANOME_AVAILABLE:
            analyzer = tokenize_ja
        self.vectorizer = TfidfVectorizer(max_features=4000, tokenizer=analyzer)
        self.question_vectors = self.vectorizer.fit_transform(self.df['Question'])
        self.answer_vectors = TfidfVectorizer(max_features=4000, tokenizer=analyzer).fit_transform(self.df['Answer'])

        # 埋め込み（任意）
        self.use_embeddings = use_embeddings and _EMBED_AVAILABLE
        self.embed_model = None
        self.q_embed = None
        self.a_embed = None
        if self.use_embeddings:
            self.embed_model = SentenceTransformer(embed_model_name)
            self.q_embed = self.embed_model.encode(self.df['Question'].tolist(), normalize_embeddings=True)
            self.a_embed = self.embed_model.encode(self.df['Answer'].tolist(), normalize_embeddings=True)

        # 大谷選手の発言パターンを抽出
        self.speech_patterns = self._extract_speech_patterns()
        # 大谷選手の答え方の癖を学習
        self.style = self._learn_style()

    def _extract_speech_patterns(self) -> List[str]:
        """大谷選手の実際の発言からパターンを抽出"""
        patterns = []
        answers = self.df['Answer'].tolist()
        common_patterns = [
            r'まだまだ.*?と思います',
            r'.*?のおかげで.*?',
            r'.*?から学ぶことが.*?',
            r'そうですね.*?',
            r'.*?だと思うので.*?',
            r'.*?というのは.*?',
            r'.*?かなと思います',
            r'.*?じゃないかなと.*?',
            r'.*?していきたいなと思います',
            r'.*?ということです',
        ]
        for answer in answers:
            for pattern in common_patterns:
                matches = re.findall(pattern, answer)
                patterns.extend(matches[:3])
        return list(set(patterns))[:50]

    def _learn_style(self) -> Dict[str, List[str]]:
        """Answer列から『答え方の癖』を学習する簡易スタイルモデル"""
        answers = [str(a) for a in self.df['Answer'].tolist() if isinstance(a, str)]
        starters_count = {}
        endings_count = {}
        hesitations = ["そうですね", "んー", "まあ", "あの", "やっぱり"]
        connectors = ["まず", "その上で", "なので", "ただ", "一方で", "しっかり", "徐々に"]
        for a in answers:
            # 文ごと
            sents = [s for s in re.split(r"[。!?！？」]", a) if s.strip()]
            if sents:
                start = sents[0].strip()[:6]
                starters_count[start] = starters_count.get(start, 0) + 1
            # 代表的な文末
            m = re.findall(r"(と思います|かなと思います|だと思います|じゃないかなと|ではないかな|していきたい|です|ます)\s*$", a)
            for e in m:
                endings_count[e] = endings_count.get(e, 0) + 1
        # 上位抽出
        def topk(d: Dict[str, int], k: int) -> List[str]:
            return [w for w, _ in sorted(d.items(), key=lambda x: x[1], reverse=True)[:k]]
        style = {
            "starters": topk(starters_count, 10) or ["そうですね"],
            "endings": topk(endings_count, 10) or ["と思います"],
            "hesitations": hesitations,
            "connectors": connectors,
        }
        return style

    def _apply_style(self, query: str, core_text: str | None = None) -> str:
        """学習した癖を使って応答を包む"""
        import random
        starter = random.choice(self.style["starters"]) if self.style["starters"] else "そうですね"
        hesi = random.choice(self.style["hesitations"]) if np.random.rand() < 0.6 else ""
        conn = random.choice(self.style["connectors"]) if np.random.rand() < 0.5 else ""
        ending = random.choice(self.style["endings"]) if self.style["endings"] else "と思います"
        prefix = f"{starter}、" if starter else ""
        if hesi and not prefix.startswith(hesi):
            prefix = hesi + "、" + prefix
        body = core_text.strip() if core_text else f"{query}については{ending}。"
        if not body.endswith("。"):
            body += "。"
        tail = f"{conn}、これからもしっかり準備していきたい{ending}。" if conn else f"これからも継続していきたい{ending}。"
        return prefix + body + tail

    def _similarities(self, query: str, use_answer_space: bool = False) -> np.ndarray:
        """埋め込み or TF-IDF で類似度を返す"""
        if self.use_embeddings:
            vec = self.embed_model.encode([query], normalize_embeddings=True)[0]
            mat = self.a_embed if use_answer_space else self.q_embed
            return cosine_similarity(vec.reshape(1, -1), mat)[0]
        else:
            vectorizer = (TfidfVectorizer(max_features=4000, tokenizer=tokenize_ja)
                         if _JANOME_AVAILABLE else self.vectorizer)
            vec = vectorizer.fit(self.df['Question']).transform([query])
            mat = self.answer_vectors if use_answer_space else self.question_vectors
            return cosine_similarity(vec, mat)[0]

    def layer1_direct_search(self, query: str, threshold: float = 0.7, top_k: int = 1, mmr_lambda: float = 0.5) -> Tuple[List[Dict], str]:
        """Layer 1: 直接検索 → 信頼度: 高"""
        sims = self._similarities(query, use_answer_space=False)
        if top_k == 1:
            best_idx = int(np.argmax(sims))
            best_score = float(sims[best_idx])
            if best_score >= threshold:
                return [{
                    'question': self.df.iloc[best_idx]['Question'],
                    'answer': self.df.iloc[best_idx]['Answer'],
                    'score': best_score,
                    'id': int(self.df.iloc[best_idx]['ID']),
                    'confidence': 'high'
                }], "直接検索で高精度マッチを発見"
            return [], "直接検索では該当なし"
        # MMRで複数取得
        idxs = mmr(sims.reshape(-1, 1), sims.reshape(-1, 1), top_k=top_k, lambda_mult=mmr_lambda)
        results = []
        for i in idxs:
            if sims[i] >= threshold:
                results.append({
                    'question': self.df.iloc[i]['Question'],
                    'answer': self.df.iloc[i]['Answer'],
                    'score': float(sims[i]),
                    'id': int(self.df.iloc[i]['ID']),
                    'confidence': 'high'
                })
        return (results, f"直接検索で{len(results)}件") if results else ([], "直接検索では該当なし")

    def layer2_concept_search(self, query: str, threshold: float = 0.5, top_k: int = 3, mmr_lambda: float = 0.5) -> Tuple[List[Dict], str]:
        """Layer 2: 概念検索 → 信頼度: 中"""
        concept_expansions = {
            'AI': ['技術', '新しい', '学習', '進歩', '挑戦', '未来'],
            '宇宙': ['夢', '挑戦', '新しい', '目標', '広い'],
            '料理': ['食事', '好き', '楽しい', 'おいしい'],
            '映画': ['見る', '楽しい', '時間', 'リラックス'],
            '音楽': ['聞く', '好き', 'リラックス', '楽しい'],
            '家族': ['大切', '支え', 'ありがたい', '感謝'],
            '友達': ['仲間', 'チームメート', '大切', '信頼'],
            '将来': ['目標', '夢', '挑戦', '頑張る'],
            '困難': ['挑戦', '乗り越える', '学ぶ', '成長'],
            '成功': ['努力', '継続', 'チーム', '感謝'],
        }
        expanded_query = query
        for concept, keywords in concept_expansions.items():
            if concept in query:
                expanded_query += ' ' + ' '.join(keywords)
        sims = self._similarities(expanded_query, use_answer_space=False)
        idxs = np.argsort(sims)[-top_k:][::-1]
        results = []
        for idx in idxs:
            if sims[idx] >= threshold:
                results.append({
                    'question': self.df.iloc[idx]['Question'],
                    'answer': self.df.iloc[idx]['Answer'],
                    'score': float(sims[idx]),
                    'id': int(self.df.iloc[idx]['ID']),
                    'confidence': 'medium'
                })
        return (results, f"概念検索で{len(results)}件発見 (拡張: {expanded_query})") if results else ([], "概念検索でも該当なし")

    def layer3_pattern_generation(self, query: str) -> Tuple[str, str]:
        """Layer 3: パターン生成 → 信頼度: 低"""
        import random
        selected_patterns = random.sample(self.speech_patterns, min(3, len(self.speech_patterns)))
        # 回答空間に対する近傍からキーワードを拝借
        sims = self._similarities(query, use_answer_space=True)
        top_idx = int(np.argmax(sims))
        related_answer = self.df.iloc[top_idx]['Answer']
        generated_response = self._generate_pattern_response(query, selected_patterns, related_answer)
        return generated_response, f"パターン生成 (使用パターン: {len(selected_patterns)}個)"

    def _generate_pattern_response(self, query: str, patterns: List[str], related_answer: str) -> str:
        # 既存の組み立てに学習スタイルを適用
        base_responses = [
            f"{query}に関しては",
            f"うーん、{query}というのは",
            f"{query}については",
        ]
        middle_parts = [
            "新しいことに挑戦するのは素晴らしいことだと思います",
            "まだまだ学ぶことがたくさんあると感じています",
            "これからも努力を続けていきたいと思います",
            "チームのみんなのおかげで成長できています",
        ]
        ending_parts = [
            "野球以外のことからも学ぶことがたくさんあると思っています。",
            "これからも頑張っていきたいなと思います。",
            "継続していくことが大切だと思います。",
        ]
        import random
        core = random.choice(base_responses) + random.choice(middle_parts) + "。" + random.choice(ending_parts)
        return self._apply_style(query, core)

    def aggregate_answers(self, hits: List[Dict], max_chars: int = 400, style: bool = False, query: str = "") -> str:
        """複数根拠を短く統合（任意でスタイル適用）"""
        if not hits:
            return ""
        texts = []
        for h in hits:
            ans = str(h['answer']).strip()
            if ans:
                texts.append(ans)
        core = " ".join(texts)[:max_chars]
        return self._apply_style(query, core) if style else core

    def search(self, query: str, l1_th: float, l2_th: float, top_k: int, mmr_lambda: float, style_on_aggregate: bool) -> Dict:
        """3層段階検索の実行（設定反映）"""
        # Layer 1
        results1, msg1 = self.layer1_direct_search(query, threshold=l1_th, top_k=top_k, mmr_lambda=mmr_lambda)
        if results1:
            answer = results1[0]['answer'] if top_k == 1 else self.aggregate_answers(results1, style=style_on_aggregate, query=query)
            return {
                'layer': 1,
                'results': results1,
                'message': msg1,
                'confidence': 'high',
                'response': answer,
                'source': f"; ".join([f"ID {r['id']}: {r['question'][:40]}" for r in results1])
            }

        # Layer 2
        results2, msg2 = self.layer2_concept_search(query, threshold=l2_th, top_k=top_k, mmr_lambda=mmr_lambda)
        if results2:
            answer = results2[0]['answer'] if top_k == 1 else self.aggregate_answers(results2, style=style_on_aggregate, query=query)
            return {
                'layer': 2,
                'results': results2,
                'message': msg2,
                'confidence': 'medium',
                'response': answer,
                'source': f"; ".join([f"ID {r['id']}: {r['question'][:40]}" for r in results2])
            }

        # Layer 3
        generated_response, msg3 = self.layer3_pattern_generation(query)
        return {
            'layer': 3,
            'results': [],
            'message': msg3,
            'confidence': 'low',
            'response': generated_response,
            'source': "大谷選手の発言パターンから生成"
        }


def main():
    st.title("⚾ 大谷翔平 AI Chat")
    st.subheader("🚀 3層段階RAGシステム - 真のRAG価値を発揮するハイブリッドシステム")

    # サイドバー設定
    st.sidebar.header("⚙️ 設定")
    use_embeddings = st.sidebar.checkbox("Sentence-Transformers 埋め込みを使う", value=_EMBED_AVAILABLE)
    embed_model = st.sidebar.text_input("埋め込みモデル", value="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    top_k = st.sidebar.slider("Top K (根拠件数)", 1, 5, 1)
    l1_th = st.sidebar.slider("Layer1 閾値", 0.0, 1.0, 0.7, 0.05)
    l2_th = st.sidebar.slider("Layer2 閾値", 0.0, 1.0, 0.5, 0.05)
    mmr_lambda = st.sidebar.slider("MMR λ (多様性)", 0.0, 1.0, 0.5, 0.05)
    style_on_aggregate = st.sidebar.checkbox("複数根拠を大谷スタイルで要約する", value=True)

    # RAGシステムの初期化
    @st.cache_resource
    def load_rag_system_cached(_use_embeddings: bool, _model: str):
        return OhtaniRAGSystem('ohtani_rag_final.csv', _use_embeddings, _model)
    try:
        rag_system = load_rag_system_cached(use_embeddings, embed_model)
        st.success(f"✅ RAG初期化完了 ({len(rag_system.df)}件 / embeddings={rag_system.use_embeddings})")
    except Exception as e:
        st.error(f"❌ 初期化エラー: {e}")
        return

    st.markdown("---")
    query = st.text_input(
        "💬 大谷選手に質問してください:",
        placeholder="例: 野球以外で興味のあることはありますか？",
        help="どんな質問でも3層システムが適切な回答を見つけます"
    )

    if st.button("🔍 質問する", type="primary"):
        if query.strip():
            with st.spinner("🤖 3層段階検索を実行中..."):
                result = rag_system.search(query, l1_th=l1_th, l2_th=l2_th, top_k=top_k, mmr_lambda=mmr_lambda, style_on_aggregate=style_on_aggregate)

            st.markdown("---")
            confidence_colors = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
            confidence_labels = {'high': '高', 'medium': '中', 'low': '低'}
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                st.metric("使用レイヤー", f"Layer {result['layer']}")
            with col2:
                st.metric("信頼度", f"{confidence_colors[result['confidence']]} {confidence_labels[result['confidence']]}")
            with col3:
                st.info(result['message'])

            st.markdown("### 💬 大谷選手の回答")
            st.markdown(f"**{result['response']}**")

            st.markdown("### 📝 参考情報")
            st.markdown(f"**出典:** {result['source']}")
        else:
            st.warning("質問を入力してください。")

    st.markdown("---")
    st.markdown("### 💡 サンプル質問")
    sample_questions = [
        "野球以外で興味のあることはありますか？",
        "宇宙旅行についてどう思いますか？",
        "AIについての考えを聞かせてください",
        "好きな食べ物は何ですか？",
        "将来の夢について教えてください",
        "困難を乗り越える秘訣は？"
    ]
    cols = st.columns(2)
    for i, q in enumerate(sample_questions):
        with cols[i % 2]:
            if st.button(f"📋 {q}", key=f"sample_{i}"):
                st.session_state.sample_query = q
                st.rerun()

    if 'sample_query' in st.session_state:
        query = st.session_state.sample_query
        del st.session_state.sample_query
        with st.spinner("🤖 3層段階検索を実行中..."):
            result = rag_system.search(query, l1_th=l1_th, l2_th=l2_th, top_k=top_k, mmr_lambda=mmr_lambda, style_on_aggregate=style_on_aggregate)
        st.markdown("---")
        confidence_colors = {'high': '🔴', 'medium': '🟡', 'low': '🟢'}
        confidence_labels = {'high': '高', 'medium': '中', 'low': '低'}
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.metric("使用レイヤー", f"Layer {result['layer']}")
        with col2:
            st.metric("信頼度", f"{confidence_colors[result['confidence']]} {confidence_labels[result['confidence']]}")
        with col3:
            st.info(result['message'])
        st.markdown("### 💬 大谷選手の回答")
        st.markdown(f"**{result['response']}**")
        st.markdown("### 📝 参考情報")
        st.markdown(f"**出典:** {result['source']}")

if __name__ == "__main__":
    main()