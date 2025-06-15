#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
한국어 지원 감정 분석 및 사이버 폭력 위험도 예측 시스템
- 한국어 전용
- 감정 분류: 긍정, 부정, 중립
- 사이버 폭력 위험도: 높음, 보통, 낮음
"""

import pandas as pd
import numpy as np
import re
import warnings
import os # 현재는 사용되지 않으나, 추후 파일 로드 시 필요할 수 있으므로 유지
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Matplotlib 한글 및 음수 깨짐 방지 설정
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 기준, macOS는 'AppleGothic', Linux는 'NanumGothic' 등
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings('ignore')

# 한국어 처리 라이브러리 로드 (필수)
try:
    from konlpy.tag import Okt # Mecab, Komoran 등 다른 형태소 분석기 사용 가능
    from soynlp.normalizer import repeat_normalize
    print("한국어 처리 라이브러리 (Konlpy, Soynlp) 로드 완료")
except ImportError:
    raise ImportError("한국어 처리 라이브러리(konlpy, soynlp)가 설치되지 않았습니다. 'pip install konlpy soynlp'를 실행해주세요.")

class KoreanEmotionCyberbullyingAnalyzer:
    def __init__(self):
        self.emotion_model = None
        self.cyberbullying_model = None
        self.vectorizer_emotion = None
        self.vectorizer_cyberbullying = None
        
        # 한국어 형태소 분석기 초기화
        self.korean_tokenizer = Okt()
        # 한국어 불용어 (추가 필요 시 확장)
        self.korean_stopwords = [
            '은', '는', '이', '가', '을', '를', '에', '의', '로', '와', '과', 
            '도', '만', '에서', '부터', '까지', '보다', '처럼', '같이', '하다',
            '이다', '되다', '아니다', '그리고', '또는', '하지만', '그러나', '그래서', '아', '휴'
        ]
        
        # 한국어 감정 단어 사전 (확장 필요)
        self.korean_emotion_dict = {
            'positive': ['좋다', '행복하다', '기쁘다', '사랑하다', '훌륭하다', '완벽하다', '최고', '대박',
                         '멋있다', '예쁘다', '아름답다', '감사하다', '고맙다', '신나다', '즐겁다', '웃다',
                         '만족하다', '성공하다', '승리하다', '축하하다', '좋아하다', '괜찮다', '감동', '편안'],
            'negative': ['나쁘다', '싫다', '화나다', '슬프다', '우울하다', '짜증나다', '답답하다', '실망하다',
                         '후회하다', '걱정하다', '무섭다', '두렵다', '어렵다', '힘들다', '아프다', '괴롭다',
                         '미치다', '죽겠다', '끔찍하다', '최악', '망하다', '실패하다', '혐오', '불안']
        }
        
        # 한국어 혐오 표현 사전 (확장 필요)
        self.korean_hate_dict = {
            'high_risk': ['죽어', '꺼져', '한국에서나가', '바퀴벌레', '쓰레기', '개새끼', '미친놈', '병신',
                          '멍청이', '바보', '너같은놈', '역겨워', '토나와', '혐오스러워', '싫어죽겠어', '재기해'],
            'medium_risk': ['답답해', '짜증나', '웃기네', '황당해', '어이없어', '멍청해', '바보같아', '시끄러', '꼴불견'],
            'profanity': ['시발', '씨발', '개소리', '좆', '니미', '새끼', '년', '놈', '개', '존나', '씹']
        }

    def preprocess_text(self, text):
        """한국어 텍스트 전처리"""
        if pd.isna(text):
            return ""
        
        # 반복 문자 정규화 (예: ㅋㅋㅋㅋㅋ -> ㅋㅋ, 너무너무너무 -> 너무너무)
        text = repeat_normalize(text, num_repeats=2)
        
        # 특수문자 및 숫자 제거 (한글과 공백만 유지)
        text = re.sub(r'[^가-힣\s]', '', text)
        
        # 형태소 분석
        # nouns(), morphs(), pos() 등 사용 가능. 여기서는 morphs (어간/어미/조사 등 구분X)
        # 더 정확한 분석을 위해 pos()를 사용하고 명사/동사/형용사 등만 추출하는 방식도 고려
        tokens = self.korean_tokenizer.morphs(text) 
        
        # 불용어 제거 및 한 글자 단어 제거
        tokens = [token for token in tokens 
                  if token not in self.korean_stopwords and len(token) > 1]
        
        return ' '.join(tokens)

    def load_sample_datasets(self):
        """한국어 샘플 데이터셋 로드 및 확장"""
        datasets = {}
        
        # 한국어 감정 분석 데이터셋 (샘플, 양 확장)
        emotion_data_ko = {
            'text': [
                "이 제품 정말 좋아요! 강력 추천합니다", # 긍정
                "최악의 서비스네요, 다시는 이용 안할거예요", # 부정
                "그냥 그래요, 보통 수준이에요", # 중립
                "정말 훌륭한 품질이네요, 만족합니다", # 긍정
                "너무 실망스러워요, 돈 아까워요", # 부정
                "무난해요, 나쁘지 않아요", # 중립
                "대박이에요! 완전 좋아요", # 긍정
                "별로예요, 기대했는데 실망이에요", # 부정
                "괜찮은 편이에요, 평범해요", # 중립
                "가격 대비 훌륭해요, 추천!", # 긍정
                "너무 행복하고 즐거운 하루였어요.", # 긍정
                "정말 짜증나고 화가 나네요.", # 부정
                "특별할 것 없는 평범한 일상입니다.", # 중립
                "최고의 선택이었고, 전혀 후회하지 않아요.", # 긍정
                "기대 이하였어요. 정말 실망스럽네요." # 부정
            ] * 200, # 3000개 샘플로 확장 (원래 10개 * 30 = 300)
            'emotion': ['positive', 'negative', 'neutral', 'positive', 'negative', 
                        'neutral', 'positive', 'negative', 'neutral', 'positive',
                        'positive', 'negative', 'neutral', 'positive', 'negative'] * 200
        }
        
        # 한국어 사이버 폭력 데이터셋 (샘플, 양 확장)
        cyberbully_data_ko = {
            'text': [
                "너 진짜 바보같다, 그냥 죽어버려", # 높음
                "의견 공유해주셔서 감사합니다", # 낮음
                "멍청한 놈, 꺼져버려", # 높음
                "좋은 지적이네요, 고마워요", # 낮음
                "니같은 쓰레기는 세상에 필요없어", # 높음
                "더 건설적으로 이야기해봐요", # 낮음
                "병신같은 소리 하지마", # 보통
                "그런 관점도 있군요", # 낮음
                "짜증나게 하지마, 답답해 죽겠어", # 보통
                "흥미로운 의견이네요", # 낮음
                "존나 싫으니까 눈에 띄지마", # 높음
                "그냥 조용히 해줄 수 없어?", # 보통
                "너 때문에 내가 망했어", # 보통
                "정신병원에나 가봐", # 높음
                "너 같은 애들은 사회의 암덩어리야", # 높음
                "괜찮습니다.", # 낮음
                "아, 진짜 개빡치네", # 보통
                "개새끼", # 높음
                "제발 그만해", # 낮음
                "못생겼으면 조용히 있어" # 높음
            ] * 150, # 2000개 샘플로 확장 (원래 10개 * 25 = 250)
            'risk_level': ['high', 'low', 'high', 'low', 'high', 
                           'low', 'medium', 'low', 'medium', 'low',
                           'high', 'medium', 'medium', 'high', 'high',
                           'low', 'medium', 'high', 'low', 'high'] * 150
        }
        
        datasets['emotion_ko'] = pd.DataFrame(emotion_data_ko)
        datasets['cyberbully_ko'] = pd.DataFrame(cyberbully_data_ko)
        
        return datasets

    def extract_features(self, text):
        """규칙 기반 특성 추출 (한국어 전용)"""
        features = {}
        text_lower = text.lower()
        
        emotion_dict = self.korean_emotion_dict
        hate_dict = self.korean_hate_dict
        
        # 감정 단어 빈도
        positive_count = sum(1 for word in emotion_dict['positive'] if word in text_lower)
        negative_count = sum(1 for word in emotion_dict['negative'] if word in text_lower)
        
        # 혐오 표현 빈도
        high_risk_count = sum(1 for word in hate_dict['high_risk'] if word in text_lower)
        medium_risk_count = sum(1 for word in hate_dict['medium_risk'] if word in text_lower)
        profanity_count = sum(1 for word in hate_dict['profanity'] if word in text_lower)
        
        features.update({
            'positive_words': positive_count,
            'negative_words': negative_count,
            'high_risk_words': high_risk_count,
            'medium_risk_words': medium_risk_count,
            'profanity_count': profanity_count,
            'text_length': len(text),
            'word_count': len(text.split()),
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'caps_ratio': sum(1 for c in text if '가' <= c <= '힣' and c.isupper()) / len(text) if text else 0 # 한국어 대문자는 거의 없음
        })
        
        return features

    def train_models(self, datasets):
        """한국어 모델 훈련"""
        print("한국어 감정 분석 및 사이버 폭력 예측 모델 훈련 중...")
        
        emotion_df = datasets['emotion_ko']
        cyberbully_df = datasets['cyberbully_ko']
        
        print(f"감정 분석 데이터: {len(emotion_df)}개")
        print(f"사이버 폭력 데이터: {len(cyberbully_df)}개")
        
        # 감정 분석 모델 훈련
        emotion_df['processed_text'] = emotion_df['text'].apply(self.preprocess_text)
        X_train_e, X_test_e, y_train_e, y_test_e = train_test_split(
            emotion_df['processed_text'], emotion_df['emotion'],
            test_size=0.2, random_state=42, stratify=emotion_df['emotion']
        )
        
        self.vectorizer_emotion = TfidfVectorizer(
            max_features=3000, # 특성 개수 증가
            ngram_range=(1, 3), # 1-gram, 2-gram, 3-gram 모두 고려
            min_df=2,
            max_df=0.9
        )
        
        X_train_e_vec = self.vectorizer_emotion.fit_transform(X_train_e)
        X_test_e_vec = self.vectorizer_emotion.transform(X_test_e)
        
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=2000), # max_iter 증가
            'Naive Bayes': MultinomialNB(),
            'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced') # n_estimators 증가, 불균형 데이터 대비
        }
        
        best_emotion_model = None
        best_emotion_score = 0
        
        print("\n=== 감정 분석 모델 성능 ===")
        for name, model in models.items():
            model.fit(X_train_e_vec, y_train_e)
            score = model.score(X_test_e_vec, y_test_e)
            print(f"{name}: {score:.4f}")
            
            if score > best_emotion_score:
                best_emotion_score = score
                best_emotion_model = model
        
        self.emotion_model = best_emotion_model
        print(f"\n최적 감정 분석 모델: {type(best_emotion_model).__name__} (정확도: {best_emotion_score:.4f})")
        print("\n감정 분석 모델 분류 보고서:")
        print(classification_report(y_test_e, best_emotion_model.predict(X_test_e_vec)))

        # 사이버 폭력 모델 훈련
        cyberbully_df['processed_text'] = cyberbully_df['text'].apply(self.preprocess_text)
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            cyberbully_df['processed_text'], cyberbully_df['risk_level'],
            test_size=0.2, random_state=42, stratify=cyberbully_df['risk_level']
        )
        
        self.vectorizer_cyberbullying = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.9
        )
        
        X_train_c_vec = self.vectorizer_cyberbullying.fit_transform(X_train_c)
        X_test_c_vec = self.vectorizer_cyberbullying.transform(X_test_c)
        
        best_cyber_model = None
        best_cyber_score = 0
        
        print("\n=== 사이버 폭력 예측 모델 성능 ===")
        for name, model in models.items(): # 같은 모델 셋 사용
            # 모델 인스턴스 복사 (각 모델이 독립적으로 훈련되도록)
            model_copy = models[name].__class__(**models[name].get_params())
            model_copy.fit(X_train_c_vec, y_train_c)
            score = model_copy.score(X_test_c_vec, y_test_c)
            print(f"{name}: {score:.4f}")
            
            if score > best_cyber_score:
                best_cyber_score = score
                best_cyber_model = model_copy
        
        self.cyberbullying_model = best_cyber_model
        print(f"\n최적 사이버 폭력 예측 모델: {type(best_cyber_model).__name__} (정확도: {best_cyber_score:.4f})")
        print("\n사이버 폭력 예측 모델 분류 보고서:")
        print(classification_report(y_test_c, best_cyber_model.predict(X_test_c_vec)))

        # 반환값 변경: 테스트 데이터, 실제값, 예측값
        return (X_test_e, y_test_e, self.emotion_model.predict(X_test_e_vec)), \
               (X_test_c, y_test_c, self.cyberbullying_model.predict(X_test_c_vec))

    def analyze_text(self, text):
        """한국어 텍스트 통합 분석"""
        features = self.extract_features(text)
        
        print(f"분석할 텍스트: '{text}'")
        print("-" * 60)
        
        # 머신러닝 기반 예측
        if self.emotion_model and self.cyberbullying_model:
            processed_text = self.preprocess_text(text)
            
            # 감정 분석
            emotion_vec = self.vectorizer_emotion.transform([processed_text])
            emotion_pred = self.emotion_model.predict(emotion_vec)[0]
            emotion_proba = self.emotion_model.predict_proba(emotion_vec)[0]
            
            # 사이버 폭력 위험도 예측
            cyber_vec = self.vectorizer_cyberbullying.transform([processed_text])
            cyber_pred = self.cyberbullying_model.predict(cyber_vec)[0]
            cyber_proba = self.cyberbullying_model.predict_proba(cyber_vec)[0]
            
            print("=== 머신러닝 예측 결과 ===")
            print(f"감정: {emotion_pred}")
            # 클래스 순서를 정렬하여 출력의 일관성 확보
            emotion_classes_sorted = sorted(self.emotion_model.classes_)
            print(f"감정 확률: {{'{emotion_classes_sorted[0]}': {emotion_proba[0]:.4f}, '{emotion_classes_sorted[1]}': {emotion_proba[1]:.4f}, '{emotion_classes_sorted[2]}': {emotion_proba[2]:.4f}}}")
            
            print(f"사이버 폭력 위험도: {cyber_pred}")
            cyber_classes_sorted = sorted(self.cyberbullying_model.classes_)
            print(f"위험도 확률: {{'{cyber_classes_sorted[0]}': {cyber_proba[0]:.4f}, '{cyber_classes_sorted[1]}': {cyber_proba[1]:.4f}, '{cyber_classes_sorted[2]}': {cyber_proba[2]:.4f}}}")
        
        # 규칙 기반 분석
        emotion_rule = self.rule_based_emotion_analysis(text)
        cyber_rule = self.rule_based_cyberbullying_analysis(text)
        
        print("\n=== 규칙 기반 분석 결과 ===")
        print(f"감정: {emotion_rule}")
        print(f"사이버 폭력 위험도: {cyber_rule}")
        
        print(f"\n=== 텍스트 특성 ===")
        for key, value in features.items():
            print(f"{key}: {value}")
        
        print("=" * 60)

    def rule_based_emotion_analysis(self, text):
        """규칙 기반 감정 분석 (한국어 전용)"""
        emotion_dict = self.korean_emotion_dict
        
        text_lower = text.lower()
        positive_score = sum(1 for word in emotion_dict['positive'] if word in text_lower)
        negative_score = sum(1 for word in emotion_dict['negative'] if word in text_lower)
        
        if positive_score > negative_score and positive_score > 0:
            return 'positive'
        elif negative_score > positive_score and negative_score > 0:
            return 'negative'
        else:
            return 'neutral'

    def rule_based_cyberbullying_analysis(self, text):
        """규칙 기반 사이버 폭력 위험도 분석 (한국어 전용)"""
        hate_dict = self.korean_hate_dict
        
        text_lower = text.lower()
        high_risk_score = sum(1 for word in hate_dict['high_risk'] if word in text_lower)
        medium_risk_score = sum(1 for word in hate_dict['medium_risk'] if word in text_lower)
        profanity_score = sum(1 for word in hate_dict['profanity'] if word in text_lower)
        
        total_risk_score = high_risk_score * 3 + medium_risk_score * 2 + profanity_score * 1.5
        
        # 임계값 조정 (데이터 특성에 따라 최적화 필요)
        if total_risk_score >= 4:
            return 'high'
        elif total_risk_score >= 1.5:
            return 'medium'
        else:
            return 'low'

    def plot_confusion_matrices(self, emotion_results, cyberbullying_results):
        """혼동 행렬 시각화"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 감정 분석 혼동 행렬
        cm_emotion = confusion_matrix(emotion_results[1], emotion_results[2])
        # 클래스 레이블 순서 정렬 (일관성 위해)
        emotion_labels = sorted(list(set(emotion_results[1])))
        sns.heatmap(cm_emotion, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=emotion_labels,
                    yticklabels=emotion_labels,
                    ax=axes[0])
        axes[0].set_title('감정 분석 혼동 행렬')
        axes[0].set_xlabel('예측값')
        axes[0].set_ylabel('실제값')
        
        # 사이버 폭력 위험도 혼동 행렬
        cm_cyber = confusion_matrix(cyberbullying_results[1], cyberbullying_results[2])
        # 클래스 레이블 순서 정렬 (일관성 위해)
        cyber_labels = sorted(list(set(cyberbullying_results[1])))
        sns.heatmap(cm_cyber, annot=True, fmt='d', cmap='Reds',
                    xticklabels=cyber_labels,
                    yticklabels=cyber_labels,
                    ax=axes[1])
        axes[1].set_title('사이버 폭력 위험도 혼동 행렬')
        axes[1].set_xlabel('예측값')
        axes[1].set_ylabel('실제값')
        
        plt.tight_layout()
        plt.show()

def main():
    """메인 실행 함수"""
    print("한국어 감정 분석 및 사이버 폭력 위험도 예측 시스템")
    print("=" * 60)
    
    # 분석기 초기화
    analyzer = KoreanEmotionCyberbullyingAnalyzer()
    
    # 샘플 데이터셋 로드
    print("샘플 데이터셋 로드 중...")
    datasets = analyzer.load_sample_datasets()
    
    for name, df in datasets.items():
        print(f"{name}: {len(df)}개 샘플")
    
    # 모델 훈련
    print("\n모델 훈련 시작...")
    emotion_results, cyber_results = analyzer.train_models(datasets)
    
    # 테스트 문장들 (한국어 전용)
    test_texts = [
        "이 영화 정말 최고예요! 완전 감동",
        "진짜 짜증나고 기분 더러워, 꺼져버려!",
        "오늘 날씨는 그냥 그렇네요.",
        "너 같은 애들은 진짜 답이 없어. 멍청이.",
        "감사합니다. 큰 도움이 됐어요.",
        "하... 병신 같은 소리 하고 있네.",
        "회의가 아주 지루했어요.",
        "이거 정말 완벽한 해결책이에요!"
    ]
    
    print("\n" + "=" * 60)
    print("테스트 문장 분석 결과")
    print("=" * 60)
    
    for text in test_texts:
        analyzer.analyze_text(text)
        print()
    
    # 혼동 행렬 시각화
    try:
        analyzer.plot_confusion_matrices(emotion_results, cyber_results)
    except Exception as e:
        print(f"시각화 생성 중 오류: {e}")
    
    # 대화형 모드
    print("\n대화형 분석 모드 (종료하려면 'quit' 또는 '종료' 입력)")
    
    while True:
        user_input = input("\n분석할 한국어 텍스트를 입력하세요: ")
        if user_input.lower() in ['quit', 'exit', '종료']:
            break
        if user_input.strip():
            analyzer.analyze_text(user_input)

if __name__ == "__main__":
    main()

