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
from datetime import datetime
import html
import textwrap

# ページナビゲーション（簡易版）
def show_page_navigation() -> str:
    """現在はチャットページ固定で返す。必要ならタブやラジオUIに拡張可。"""
    return 'chat'

def show_home_page():
    """ホームページ（現状はチャットページに委譲）。"""
    show_chat_page()

def show_extraction_page():
    """抽出ページのプレースホルダー。"""
    st.markdown("### 抽出ページ\n準備中です。左のチャット設定からAIキーを設定して会話をお試しください。")

def show_settings_page():
    """設定ページのプレースホルダー。"""
    st.markdown("### 設定\nサイドバーの『チャット設定』でAIプロバイダとAPIキーを設定してください。")

# 設定
st.set_page_config(
    page_title="💬 大谷翔平チャット",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# カスタムCSS - LINE風スタイル
def load_css():
    st.markdown(textwrap.dedent("""
    <style>
    /* 全体の背景 */
    .main .block-container {
        max-width: 100% !important;
        padding-top: 1rem !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-bottom: 0 !important; /* 余白削除 */
    }
    
    /* チャット画面全体 */
    .chat-app {
        display: flex;
        flex-direction: column;
        height: auto;
        min-height: 50vh;
        max-width: 800px;
        margin: 0 auto;
        background: white;
        box-shadow: 0 0 20px rgba(0,0,0,0.1);
        border-radius: 15px;
        overflow: hidden;
    }
    
    /* ヘッダー */
    .chat-header {
        background: #e9ecef; /* グレー */
        color: #333;
        padding: 12px 16px;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        position: relative;
    }
    
    .status-indicator {
        position: absolute;
        right: 20px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 12px;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    
    .online-dot {
        width: 8px;
        height: 8px;
        background: #4CAF50;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* チャット背景 - LINE風 */
    .chat-background {
        flex: 1;
        background: #f8f9fa;
        background-image: 
            radial-gradient(circle at 20% 50%, rgba(120, 119, 198, 0.03) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(120, 119, 198, 0.03) 0%, transparent 50%),
            radial-gradient(circle at 40% 80%, rgba(120, 119, 198, 0.03) 0%, transparent 50%),
            repeating-linear-gradient(
                45deg,
                transparent,
                transparent 2px,
                rgba(120, 119, 198, 0.02) 2px,
                rgba(120, 119, 198, 0.02) 4px
            );
        padding: 20px;
        overflow-y: auto;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }
    
    /* ユーザーメッセージ（右側・LINE緑） */
    .user-message-container {
        display: flex;
        justify-content: flex-end;
        align-items: flex-end;
        gap: 10px;
        margin: 5px 0;
    }
    
    .user-message {
        background: #06c755;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        max-width: 70%;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        font-size: 16px;
        line-height: 1.4;
        word-wrap: break-word;
    }
    
    .user-avatar {
        width: 35px;
        height: 35px;
        border-radius: 50%;
        background: #06c755;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: bold;
        font-size: 12px;
        flex-shrink: 0;
    }
    
    /* 大谷メッセージ（左側・白） */
    .ohtani-message-container {
        display: flex;
        justify-content: flex-start;
        align-items: flex-end;
        gap: 10px;
        margin: 5px 0;
    }
    
    .ohtani-avatar {
        width: 35px;
        height: 35px;
        border-radius: 50%;
        background: linear-gradient(135deg, #E3F2FD, #BBDEFB);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        flex-shrink: 0;
        border: 2px solid #90CAF9;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .ohtani-message {
        background: white;
        color: #333;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        max-width: 70%;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        font-size: 16px;
        line-height: 1.4;
        word-wrap: break-word;
    }
    
    /* タイムスタンプ */
    .timestamp {
        font-size: 11px;
        color: #999;
        text-align: center;
        margin: 5px 0;
    }
    
    /* システムメッセージ */
    .system-message {
        background: rgba(0,0,0,0.05);
        color: #666;
        padding: 4px 8px;
        border-radius: 8px;
        text-align: center;
        margin: 5px auto 0 auto; /* 下マージンをなくす */
        max-width: 200px;
        font-size: 10px;
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* タイピング表示 */
    .typing-container {
        display: flex;
        align-items: flex-end;
        gap: 10px;
        margin: 5px 0;
    }
    
    .typing-indicator {
        background: white;
        color: #999;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        animation: typing-pulse 1.5s ease-in-out infinite;
        display: flex;
        align-items: center;
        gap: 5px;
    }
    
    @keyframes typing-pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1.0; }
        100% { opacity: 0.6; }
    }
    
    .typing-dots {
        display: flex;
        gap: 3px;
    }
    
    .typing-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #999;
        animation: typing-bounce 1.4s ease-in-out infinite;
    }
    
    .typing-dot:nth-child(1) { animation-delay: 0s; }
    .typing-dot:nth-child(2) { animation-delay: 0.2s; }
    .typing-dot:nth-child(3) { animation-delay: 0.4s; }
    
    @keyframes typing-bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    
    /* クイック返信エリア */
    .quick-replies {
        background: #f8f9fa;
        padding: 10px 15px;
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        border-top: 1px solid #e9ecef;
        min-height: 50px;
        align-items: center;
    }
    
    .quick-reply-btn {
        background: white;
        color: #666;
        border: 1px solid #e9ecef;
        padding: 6px 12px;
        border-radius: 15px;
        font-size: 13px;
        cursor: pointer;
        transition: all 0.3s ease;
        white-space: nowrap;
    }
    
    .quick-reply-btn:hover {
        background: #06c755;
        color: white;
        border-color: #06c755;
        transform: translateY(-1px);
    }
    
    /* 入力エリア */
    .input-area {
        background: #111827; /* ダーク背景 */
        padding: 10px 12px;
        display: flex;
        align-items: center;
        gap: 10px;
        border-top: 1px solid #1f2937;
        position: sticky;
        bottom: 0;
        z-index: 5;
        margin-bottom: 0 !important; /* 余白なくす */
    }
    
    /* Streamlitの入力フィールドスタイル調整 */
    .stTextInput > div > div > input {
        border-radius: 22px !important;
        border: 1px solid #374151 !important;
        background: #0b1220 !important;
        color: #e5e7eb !important;
        padding: 12px 16px !important;
        font-size: 16px !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #10b981 !important;
        box-shadow: 0 0 0 2px rgba(16, 185, 129, 0.15) !important;
    }
    
    /* ボタンスタイル */
    .stButton > button {
        background: #10b981 !important;
        color: white !important;
        border: none !important;
        border-radius: 22px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton > button:hover {
        background: #0ea5a4 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(14, 165, 164, 0.3) !important;
    }
    
    /* サイドバーのグレーボタン */
    .stButton > button[data-testid*="clear_history"], 
    .stButton > button[data-testid*="page_refresh"] {
        background: #6b7280 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-size: 14px !important;
        font-weight: 400 !important;
        width: 100% !important;
    }
    
    .stButton > button[data-testid*="clear_history"]:hover, 
    .stButton > button[data-testid*="page_refresh"]:hover {
        background: #4b5563 !important;
        transform: none !important;
        box-shadow: none !important;
    }
    
    /* サイドバー非表示 */
    .css-1d391kg {
        display: none;
    }
    
    /* レスポンシブ */
    @media (max-width: 768px) {
        .chat-app {
            max-width: 100%;
            height: auto;
            min-height: 60vh;
            border-radius: 0;
        }
        
        .user-message, .ohtani-message {
            max-width: 85%;
        }
        
        .chat-background {
        }
    }
    </style>
    """), unsafe_allow_html=True)

# 軽量テキスト検索クラス
class LightweightTextSearch:
    """軽量TF-IDF検索システム"""
    
    def __init__(self, texts: List[str], max_features: int = 2000):
        self.texts = texts
        self.max_features = max_features
        self.vocab = self._build_vocabulary()
        self.idf_vector = self._compute_idf()
    
    def _tokenize(self, text: str) -> List[str]:
        """日本語対応トークナイズ（改良版）"""
        if not isinstance(text, str):
            return []
        
        # 基本トークン化
        tokens = re.findall(r'[\w\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+', text.lower())
        
        # 重要キーワードの抽出と正規化
        normalized_tokens = []
        for token in tokens:
            if len(token) > 1:
                # 野球関連用語の正規化
                if '打撃' in token or 'バッティング' in token or '打席' in token:
                    normalized_tokens.extend(['打撃', 'バッティング', '打席'])
                elif '投球' in token or 'ピッチング' in token or '投手' in token:
                    normalized_tokens.extend(['投球', 'ピッチング', '投手'])
                elif '練習' in token or 'トレーニング' in token:
                    normalized_tokens.extend(['練習', 'トレーニング'])
                elif '初回' in token or '初めて' in token or 'はじめて' in token:
                    normalized_tokens.extend(['初回', '初めて', 'はじめて'])
                elif '屋外' in token or '外' in token or '野外' in token:
                    normalized_tokens.extend(['屋外', '外', '野外'])
                elif '感想' in token or '感じ' in token or 'どう' in token:
                    normalized_tokens.extend(['感想', '感じ', 'どう'])
                else:
                    normalized_tokens.append(token)
        
        return list(set(normalized_tokens))  # 重複除去
    
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

# キーワード検索
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

# チャット用RAGシステム
class OhtaniChatRAG:
    """大谷翔平チャット用RAGシステム"""
    
    def __init__(self, csv_path_or_data):
        if isinstance(csv_path_or_data, pd.DataFrame):
            # DataFrameが渡された場合
            self.df = csv_path_or_data
        elif isinstance(csv_path_or_data, str):
            # ファイルパスが渡された場合
            self.df = self._load_data(csv_path_or_data)
        else:
            # Noneまたはその他の場合はサンプルデータ
            self.df = self._create_chat_sample_data()
        
        self.questions = self.df['Question'].fillna('').astype(str).tolist()
        self.answers = self.df['Answer'].fillna('').astype(str).tolist()
        
        # 検索システム初期化
        self.tfidf_search = LightweightTextSearch(self.questions)
        self.keyword_search = KeywordSearch(self.questions)
        self.answer_search = KeywordSearch(self.answers)
        
        # 大谷選手の話し方パターン
        self.ohtani_patterns = self._extract_speech_patterns()
        
        # チャット用の挨拶・相槌パターン
        self.chat_patterns = self._create_chat_patterns()
    
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
            return self._create_chat_sample_data()
    
    def _create_chat_sample_data(self) -> pd.DataFrame:
        """チャット用サンプルデータ"""
        questions = [
            # 挨拶・日常会話
            'こんにちは', 'おはよう', '今日調子はどう？', 'お疲れさま',
            '元気？', '最近どう？', 'こんばんは', 'おやすみ',
            
            # 野球関連
            '今日の試合はどうだった？', 'バッティング調子どう？', 'ピッチングは？',
            'チームの雰囲気は？', '今シーズンの目標は？', 'プレッシャー感じる？',
            
            # プライベート
            '今何してる？', '好きな食べ物は？', '趣味は何？', '休日は何してる？',
            '映画は見る？', '音楽は聞く？', '日本が恋しい？', 'アメリカはどう？',
            
            # 励まし・応援
            '頑張って！', '応援してるよ！', 'ファイト！', '負けないで！',
            '素晴らしいプレーだった', 'すごいね！', '感動した！',
            
            # 質問・相談
            'どうしたらうまくなる？', '夢を叶える秘訣は？', '困った時はどうする？',
            '野球の楽しさって？', 'プロになるには？', 'アドバイスください',
        ]
        
        answers = [
            # 挨拶・日常会話の返答
            'こんにちは！今日も一日頑張りましょう！', 'おはようございます！今日もよろしくお願いします',
            '今日はとても調子がいいです！ありがとうございます', 'お疲れさまです！今日も頑張りました',
            'はい、おかげさまで元気です！', '最近は充実した日々を送れています',
            'こんばんは！今日も一日お疲れさまでした', 'おやすみなさい、良い夢を！',
            
            # 野球関連の返答
            '今日の試合はチーム一丸となって戦えました', 'バッティングは日々の積み重ねが大切ですね',
            'ピッチングでは一球一球集中しています', 'チームの雰囲気は本当に素晴らしいです',
            '今シーズンはチーム優勝を目指しています', 'プレッシャーも楽しみの一つだと思います',
            
            # プライベートの返答
            '今はリラックスしながら本を読んでいます', '和食、特にお母さんの手料理が恋しいです',
            '読書や映画鑑賞が好きですね', '休日は自然の中で過ごすことが多いです',
            'はい、時間がある時は映画を見ます', '音楽を聞いてリラックスしています',
            '日本の家族や友人が恋しいですね', 'アメリカでも多くのことを学んでいます',
            
            # 励まし・応援への返答
            'ありがとうございます！頑張ります！', '応援してくださって本当にありがとうございます！',
            'その言葉が力になります！', '皆さんの応援があるからこそです',
            'そう言ってもらえて嬉しいです', 'ありがとうございます！とても励みになります',
            '感動していただけて光栄です',
            
            # 質問・相談への返答
            '毎日の積み重ねが一番大切だと思います', '夢を持ち続けることが何より大切です',
            '困った時は基本に立ち返ることを心がけています', '野球は人と人をつなぐ素晴らしいスポーツです',
            'プロになるには、まず野球を心から楽しむことです', '常に謙虚さを忘れずに努力することが大切です',
        ]
        
        # データを同じ長さに調整
        max_len = max(len(questions), len(answers))
        while len(questions) < max_len:
            questions.extend(questions[:max_len-len(questions)])
        while len(answers) < max_len:
            answers.extend(answers[:max_len-len(answers)])
        
        return pd.DataFrame({
            'ID': range(1, len(questions) + 1),
            'Question': questions[:max_len],
            'Answer': answers[:max_len]
        })
    
    def _extract_speech_patterns(self) -> Dict:
        """チャット向け話し方パターン"""
        return {
            'greetings': {
                'morning': ['おはようございます！', 'おはよう！今日もよろしく！'],
                'day': ['こんにちは！', 'こんにちは！調子はどうですか？'],
                'evening': ['こんばんは！', 'こんばんは！お疲れさまでした'],
                'night': ['おやすみなさい！', 'おやすみ！良い夢を']
            },
            'starters': ['そうですね', 'うーん', 'あー', 'そうそう', 'なるほど', '実は'],
            'endings': ['です！', 'ですね', 'と思います', 'かな', 'よ', 'かもしれません'],
            'reactions': ['それはいいですね！', 'わかります！', 'そうなんです', 'なるほど！'],
            'emotions': ['嬉しいです', '楽しいですね', 'ありがたいです', '感謝しています'],
            'casual': ['はい', 'そう', 'うん', 'なるほど', 'わかりました', 'そうかも']
        }
    
    def _create_chat_patterns(self) -> Dict:
        """チャット特有のパターン"""
        return {
            'quick_responses': [
                'そうなんです！', 'ありがとうございます！', 'なるほど！', 
                'わかります！', 'その通りです！', 'いいですね！'
            ],
            'thinking': ['うーん...', 'そうですね...', 'どうでしょう...'],
            'agreement': ['はい！', 'そうです！', 'その通り！', '同感です！'],
            'encouragement': ['頑張って！', 'ファイト！', '応援しています！', '大丈夫！']
        }
    
    def chat_search(self, query: str, ai_provider: str = None, api_key: str = None) -> Dict:
        """チャット用検索（簡略化）"""
        
        # 挨拶の検出
        if self._is_greeting(query):
            return self._handle_greeting(query)
        
        # 短い返事の検出
        if self._is_short_response(query):
            return self._handle_short_response(query)
        
        # 通常のRAG検索
        threshold = 0.05  # 閾値を適度に調整
        
        # 1. 完全一致・高類似度優先検索
        best_match = self._find_best_match(query, threshold)
        if best_match is not None:
            idx, score, method = best_match
            return {
                'method': method,
                'response': self._make_chat_friendly(self.answers[idx]),
                'confidence': 'high' if score > 0.7 else 'medium',
                'needs_ai': False
            }
        
        # キーワード検索
        keyword_results = self.keyword_search.search(query, top_k=5)
        if keyword_results and keyword_results[0][1] >= threshold:
            idx, score = keyword_results[0]
            return {
                'method': 'キーワード検索',
                'response': self._make_chat_friendly(self.answers[idx]),
                'confidence': 'medium',
                'needs_ai': False  # RAGがあればAI生成はしない
            }
        
        # AI生成が必要
        if ai_provider and api_key:
            return {
                'method': 'AI生成',
                'response': None,
                'confidence': 'medium',
                'needs_ai': True,
                'ai_context': self._prepare_chat_ai_context(query)
            }
        
        # フォールバック
        return {
            'method': 'パターン生成',
            'response': self._generate_chat_response(query),
            'confidence': 'low',
            'needs_ai': False
        }
    
    def _is_greeting(self, query: str) -> bool:
        """挨拶かどうかを判定"""
        greetings = ['こんにちは', 'おはよう', 'こんばんは', 'はじめまして', 'よろしく', 'おやすみ']
        return any(greeting in query.lower() for greeting in greetings)
    
    def _handle_greeting(self, query: str) -> Dict:
        """挨拶への対応"""
        current_hour = datetime.now().hour
        
        if 'おはよう' in query:
            response = random.choice(self.ohtani_patterns['greetings']['morning'])
        elif 'こんばんは' in query or 'おやすみ' in query:
            response = random.choice(self.ohtani_patterns['greetings']['evening'])
        else:
            response = random.choice(self.ohtani_patterns['greetings']['day'])
        
        return {
            'method': '挨拶対応',
            'response': response,
            'confidence': 'high',
            'needs_ai': False
        }
    
    def _is_short_response(self, query: str) -> bool:
        """短い返事かどうかを判定"""
        short_patterns = ['はい', 'うん', 'そう', 'なるほど', 'わかった', 'ありがとう', 'すごい']
        return len(query) <= 10 and any(pattern in query.lower() for pattern in short_patterns)
    
    def _handle_short_response(self, query: str) -> Dict:
        """短い返事への対応"""
        if 'ありがとう' in query:
            response = 'こちらこそ、ありがとうございます！'
        elif 'すごい' in query or '素晴らしい' in query:
            response = 'そう言ってもらえて嬉しいです！'
        else:
            response = random.choice(self.chat_patterns['quick_responses'])
        
        return {
            'method': '短文対応',
            'response': response,
            'confidence': 'high',
            'needs_ai': False
        }
    
    def _make_chat_friendly(self, response: str) -> str:
        """回答を大谷選手らしい記者対応風に調整"""
        # 長い文を短縮
        if len(response) > 120:
            sentences = response.split('。')
            response = sentences[0] + '。'
        
        # 大谷選手の記者対応らしい表現に調整（「よ」は削除）
        # 基本的にはRAGデータそのままを尊重
        
        return response
    
    def _find_best_match(self, query: str, threshold: float):
        """最適なマッチを見つける（完全一致優先）"""
        query_clean = re.sub(r'[。、！？\s]+', '', query.lower())
        
        # 1. 完全一致・高類似度検索
        for i, question in enumerate(self.questions):
            question_clean = re.sub(r'[。、！？\s]+', '', question.lower())
            
            # 文字列類似度計算（編集距離ベース）
            similarity = self._string_similarity(query_clean, question_clean)
            
            # 90%以上の類似度なら最優先
            if similarity >= 0.9:
                return (i, similarity, '完全一致検索')
            
            # 80%以上なら高優先
            elif similarity >= 0.8:
                return (i, similarity, '高類似度検索')
        
        # 2. TF-IDF検索（範囲拡大・補正）
        tfidf_results = self.tfidf_search.search(query, top_k=15)  # 範囲拡大
        best_match = None
        best_score = 0
        
        # 上位15件を詳細チェック
        for idx, score in tfidf_results[:15]:
            if score >= threshold:
                question = self.questions[idx]
                
                # 文字列類似度も考慮
                question_clean = re.sub(r'[。、！？\s]+', '', question.lower())
                string_sim = self._string_similarity(query_clean, question_clean)
                
                # キーワード重複度
                query_keywords = set(re.findall(r'[\w\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+', query.lower()))
                question_keywords = set(re.findall(r'[\w\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]+', question.lower()))
                keyword_overlap = len(query_keywords & question_keywords) / max(len(query_keywords), 1)
                
                # 総合スコア（TF-IDF + 文字列類似度 + キーワード重複）
                combined_score = score * 0.4 + string_sim * 0.4 + keyword_overlap * 0.2
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_match = idx
        
        if best_match is not None and best_score > 0.3:
            return (best_match, best_score, 'TF-IDF検索')
        
        return None
    
    def _string_similarity(self, s1: str, s2: str) -> float:
        """文字列類似度計算（簡易版編集距離）"""
        if not s1 or not s2:
            return 0.0
        
        # 長い方を基準にする
        longer = s1 if len(s1) > len(s2) else s2
        shorter = s2 if len(s1) > len(s2) else s1
        
        if len(longer) == 0:
            return 1.0
        
        # 共通部分文字列の長さを計算
        common_chars = 0
        for char in shorter:
            if char in longer:
                common_chars += 1
        
        return common_chars / len(longer)
    
    def _generate_chat_response(self, query: str) -> str:
        """チャット向けパターン生成"""
        starter = random.choice(self.ohtani_patterns['starters'])
        ending = random.choice(self.ohtani_patterns['endings'])
        
        templates = [
            f"{starter}、それは面白い質問ですね！",
            f"いい質問{ending}！",
            f"{starter}、そのことについて考えてみますね",
            f"なるほど、{query}について{ending}",
        ]
        
        return random.choice(templates)
    
    def _prepare_chat_ai_context(self, query: str) -> str:
        """チャット用AI生成コンテキスト"""
        return f"""
あなたは大谷翔平選手として、記者に対応するときと同じような感じで回答してください。

【特徴】
- 絵文字は使わない（日本語の自然な表現で）
- 70-100文字程度
- 以下のような大谷翔平選手らしい口調で回答してください。
  特に一番最後の4が大切です。
    1. 謙虚さと誠実さ
        「まあ、そうですね」から始めることが多い
        「〜かなと思います」「〜だったんじゃないかなと思います」
        「特別なことではないですが」「あまり意識していないんですが」
        「個人的には」「僕の中では」
        「運も良かったと思います」「ラッキーだったと思います」
    2. 論理的で分かりやすい説明
        「〜という観点で」「〜に比べると」「〜というよりも」
        「〜という部分では」「〜という風に」「〜という感じです」
        「基本的に」「結果的に」「そのために」「〜だと思っているので」
    3. ポジティブで前向きな姿勢
        「それはもう、やるしかないです」
        「すごく〜だと思います」「もっと〜していきたいです」
        「自分のやるべきことは変わらないので」
        「そこは、もう切り替えて」「次の機会に」
        「チームが勝つことが一番なので」「できることは全部やる」「もちろん」
    4. 特徴的な文末パターン
        「〜という感じじゃないかなと思います」「〜かなと思います」
        「〜んじゃないかなと思います」

質問: {query}

大谷翔平として自然に返答:"""

# AI API呼び出し（チャット用）
def call_ai_for_chat(context: str, ai_provider: str, api_key: str) -> Optional[str]:
    """チャット用AI呼び出し"""
    try:
        if ai_provider == "Gemini":
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(
                context,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=150,
                    temperature=0.9,
                    top_p=0.95
                )
            )
            return response.text if hasattr(response, 'text') else None
            
        elif ai_provider == "OpenAI":
            headers = {'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}
            data = {
                'model': 'gpt-3.5-turbo',
                'messages': [{'role': 'user', 'content': context}],
                'max_tokens': 120,
                'temperature': 0.9
            }
            
            response = requests.post('https://api.openai.com/v1/chat/completions', 
                                   headers=headers, json=data, timeout=30)
            
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content']
            else:
                return f"API エラー: {response.status_code}"
                
    except Exception as e:
        return f"AI 生成エラー: {str(e)}"

# チャット履歴管理
def initialize_chat():
    """チャット履歴の初期化"""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        # 最初の挨拶メッセージ
        st.session_state.chat_history.append({
            'type': 'ohtani',
            'message': 'AI大谷です。チャットしなかったらしなかったで、みなさんうるさいですし、聞きたいことがあれば聞きます。',
            'timestamp': datetime.now().strftime('%H時%M分'),
            'method': '初期メッセージ'
        })

def add_message(message_type: str, message: str, method: str = ''):
    """メッセージを履歴に追加"""
    st.session_state.chat_history.append({
        'type': message_type,
        'message': message,
        'timestamp': datetime.now().strftime('%H時%M分'),
        'method': method
    })

def display_chat_messages():
    """チャット履歴を表示"""
    chat_html = '<div class="chat-background">'
    
    for i, msg in enumerate(st.session_state.chat_history):
        timestamp = msg.get("timestamp", "")
        # 既にHTMLとして整形済みの部分はそのまま、通常テキストはエスケープ
        raw_message = str(msg.get("message", ""))
        if raw_message.strip().startswith("<") and raw_message.strip().endswith(">"):
            safe_message = raw_message
        else:
            safe_message = html.escape(raw_message).replace("\n", "<br>")
        
        if msg['type'] == 'user':
            # メッセージの中身にHTMLタグを含む場合でも、チャットのHTMLは固定構造として出力
            chat_html += '<div class="user-message-container">'
            chat_html += f'<div class="user-message">{safe_message}</div>'
            chat_html += '<div class="user-avatar">YOU</div>'
            chat_html += '</div>'
            chat_html += f'<div class="timestamp">{timestamp}</div>'
        elif msg['type'] == 'ohtani':
            chat_html += '<div class="ohtani-message-container">'
            chat_html += '<div class="ohtani-avatar">🐶</div>'
            chat_html += f'<div class="ohtani-message">{safe_message}</div>'
            chat_html += '</div>'
            # 検索方法を時間の前に表示
            if msg.get("method") and msg.get("method") != '初期メッセージ':
                chat_html += f'<div class="system-message">{html.escape(str(msg.get("method", "")))}</div>'
            chat_html += f'<div class="timestamp">{timestamp}</div>'
        elif msg['type'] == 'system':
            chat_html += f'<div class="system-message">{msg["message"]}</div>'
        elif msg['type'] == 'typing':
            chat_html += f'''
            <div class="typing-container">
                <div class="ohtani-avatar">🐶</div>
                <div class="typing-indicator">
                    大谷選手が入力中
                    <div class="typing-dots">
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                        <div class="typing-dot"></div>
                    </div>
                </div>
            </div>
            '''
    
    chat_html += '</div>'
    return chat_html

def show_quick_replies():
    """クイック返信ボタン"""
    st.markdown('''
    <div class="quick-replies">
    ''', unsafe_allow_html=True)
    
    quick_questions = [
        "今日調子はどう？", "野球の話聞かせて", "好きな食べ物は？", 
        "今何してるの？", "応援してるよ！", "アドバイスください"
    ]
    
    # ボタンを横並びで表示
    cols = st.columns(len(quick_questions))
    selected_question = None
    
    for i, question in enumerate(quick_questions):
        with cols[i]:
            if st.button(question, key=f"quick_{i}", help=f"クイック送信: {question}"):
                selected_question = question
    
    st.markdown('</div>', unsafe_allow_html=True)
    return selected_question

# メイン関数
def main():
    # CSS読み込み
    load_css()
    
    # ページナビゲーション
    current_page = show_page_navigation()
    
    # 現在のページに応じてコンテンツ表示
    if current_page == 'home':
        show_home_page()
    elif current_page == 'chat':
        show_chat_page()
    elif current_page == 'extraction':
        show_extraction_page()
    elif current_page == 'settings':
        show_settings_page()

def show_chat_page():
    # CSS読み込み
    load_css()
    
    # チャット履歴初期化
    initialize_chat()
    
    # ヘッダーHTML
    header_html = textwrap.dedent('''
    <div class="chat-header">
        AI大谷とチャット
        <div class="status-indicator">
            <div class="online-dot"></div>
            オンライン
        </div>
    </div>
    ''')
    
    # サイドバー（設定）
    with st.sidebar:
        st.header("⚙️ チャット設定")
        
        # AI設定
        ai_provider = st.selectbox("🤖 AI生成", ["なし", "Gemini", "OpenAI"])
        api_key = ""
        
        if ai_provider == "Gemini":
            api_key = st.text_input("Gemini API Key", type="password", 
                                  value=os.getenv("GEMINI_API_KEY", ""))
        elif ai_provider == "OpenAI":
            api_key = st.text_input("OpenAI API Key", type="password",
                                  value=os.getenv("OPENAI_API_KEY", ""))
        
        use_ai = ai_provider != "なし" and bool(api_key)
        
        if use_ai:
            st.success(f"✅ {ai_provider} 接続中")
        else:
            st.info("💬 パターン応答モード")
        
        st.divider()
        
        # 統計情報
        total_messages = len(st.session_state.chat_history)
        user_messages = len([m for m in st.session_state.chat_history if m['type'] == 'user'])
        ohtani_messages = len([m for m in st.session_state.chat_history if m['type'] == 'ohtani'])
        
        st.metric("💬 総メッセージ数", total_messages)
        st.metric("👤 あなたの発言", user_messages)
        st.metric("🐶 大谷選手の返答", ohtani_messages)
        
        # 今日の大谷情報（楽しい要素）
        with st.expander("📊 今日の大谷選手"):
            st.write("⚾ 練習: バッティング練習完了")
            st.write("🏃 トレーニング: ランニング 5km")
            st.write("📚 勉強: 英語学習 30分")
            st.write("🐶 デコピン: お散歩済み")
            st.write("😊 今日の気分: 絶好調！")
            
        with st.expander("💡 使い方のコツ"):
            st.markdown("""
            **自然に話しかけてみて！**
            
            🗣️ **こんな話題がおすすめ:**
            - 今日の調子や気分
            - 野球のこと
            - 好きな食べ物や趣味  
            - 応援メッセージ
            - 相談や質問
            
            🤖 **AIモード (APIキー設定時):**
            - より自然で多様な会話
            - 新しい質問にも対応
            - 大谷選手らしい返答
            
            💬 **パターンモード:**
            - 基本的な会話に対応
            - 安定した返答
            - APIキー不要
            """)
        
        # 一番下にチャット操作ボタン（グレー）
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("履歴クリア", key="clear_history", help="チャット履歴をクリア"):
                st.session_state.chat_history = []
                initialize_chat()
                st.rerun()
        
        with col2:
            if st.button("ページ更新", key="page_refresh", help="ページを更新"):
                st.rerun()
    
    # RAGシステム初期化
    @st.cache_resource
    def load_chat_rag():
        return OhtaniChatRAG('ohtani_rag_final.csv')
    
    rag = load_chat_rag()
    
    # メインチャット画面（プレースホルダで常に置換描画）
    chat_container = st.empty()
    def render_chat(body_html: str):
        chat_container.markdown(f'<div class="chat-app">{header_html}{body_html}</div>', unsafe_allow_html=True)

    render_chat(display_chat_messages())
    
    # クイック返信（非表示化）
    # quick_reply = show_quick_replies()
    # if quick_reply:
    #     st.session_state.user_input = quick_reply
    #     st.rerun()
    
    # 入力エリア
    st.markdown('<div class="input-area">', unsafe_allow_html=True)
    
    # メッセージ入力
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.text_input(
            "メッセージを入力...", 
            key="message_input",
            placeholder="大谷選手に話しかけてみよう！",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("送信", type="primary", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # メッセージ送信処理
    if (send_button and user_input.strip()) or hasattr(st.session_state, 'user_input'):
        
        if hasattr(st.session_state, 'user_input'):
            user_input = st.session_state.user_input
            delattr(st.session_state, 'user_input')
        
        # ユーザーメッセージを追加
        add_message('user', user_input)
        
        # タイピング表示（同じプレースホルダを更新）
        typing_inner = textwrap.dedent('''
        <div class="typing-container">
            <div class="ohtani-avatar">🐶</div>
            <div class="typing-indicator">
                大谷選手が入力中
                <div class="typing-dots">
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                    <div class="typing-dot"></div>
                </div>
            </div>
        </div>
        ''')
        body_html = display_chat_messages()
        if body_html.strip().endswith('</div>'):
            body_html = body_html[:-6] + typing_inner + '</div>'
        render_chat(body_html)
        
        # 少し待機（リアル感演出）
        time.sleep(random.uniform(1.0, 2.0))
        
        # RAG検索・AI生成
        try:
            result = rag.chat_search(user_input, ai_provider if use_ai else None, api_key if use_ai else None)
            
            ohtani_response = result['response']
            method = result['method']
            
            # AI生成が必要な場合
            if result.get('needs_ai') and use_ai:
                ai_response = call_ai_for_chat(result['ai_context'], ai_provider, api_key)
                if ai_response and not ai_response.startswith(('API', 'AI')):
                    ohtani_response = ai_response.strip()
                    method = f"{ai_provider} AI生成"
                # 生成失敗やAPIエラー時はフォールバック
                if not ohtani_response:
                    ohtani_response = rag._generate_chat_response(user_input)
                    method = f"{method}→フォールバック"
            
            # 大谷選手の返答を追加
            add_message('ohtani', ohtani_response, method)
            
        except Exception as e:
            # エラー時の対応
            add_message('ohtani', 'すみません、ちょっと考えがまとまらなくて...😅 もう一度話しかけてもらえますか？', 'エラー対応')
        
        # タイピング表示を削除して再描画
        render_chat(display_chat_messages())
        st.rerun()
    
    # フッター情報（シンプル化）
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        chat_count = len([m for m in st.session_state.chat_history if m['type'] == 'user'])
        st.metric("会話数", f"{chat_count}回")
    
    with col2:
        st.markdown("#### 🐶 AI大谷とチャット中")
        if use_ai:
            st.success("🤖 AI強化モード")
        else:
            st.info("💬 基本モード")
    
    with col3:
        st.metric("メッセージ", len(st.session_state.chat_history))

if __name__ == "__main__":
    main()