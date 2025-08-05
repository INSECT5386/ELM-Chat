import tensorflow as tf
from tensorflow.keras.models import load_model
import sentencepiece as spm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class L2NormLayer(tf.keras.layers.Layer):
    def __init__(self, axis=1, epsilon=1e-10, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=self.axis, epsilon=self.epsilon)

    def get_config(self):
        config = super().get_config()
        config.update({
            "axis": self.axis,
            "epsilon": self.epsilon,
        })
        return config

# 하이퍼파라미터
MAX_SEQ_LEN = 128
BATCH_SIZE = 1024
SP_MODEL_PATH = "ko_bpe.model"
MODEL_SAVE_PATH = "sentence_encoder_model.keras"

# SentencePiece 로드
sp = spm.SentencePieceProcessor()
sp.load(SP_MODEL_PATH)

def sp_tokenize(texts, max_len=MAX_SEQ_LEN):
    encoded = sp.encode(texts, out_type=int)
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        encoded, maxlen=max_len, padding='post', truncating='post'
    )
    return padded

# 모델 로드 (safe_mode=False로 Lambda 레이어 문제 해결)
encoder = load_model(MODEL_SAVE_PATH, custom_objects={"L2NormLayer": L2NormLayer})

def encode_sentences(sentences, encoder):
    seqs = sp_tokenize(sentences, max_len=MAX_SEQ_LEN)
    dataset = tf.data.Dataset.from_tensor_slices(seqs).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return encoder.predict(dataset, verbose=0)

# 답변 후보 (예시)
answer_texts = [
    "sklearn은 머신러닝 라이브러리입니다.",
    "환불은 고객센터에 문의해주세요.",
    "주문 취소는 마이페이지에서 가능합니다.",
    "교환 문의는 고객센터에 연락 바랍니다.",
    "고객센터 연결은 1234-5678로 전화주세요."
]

# 답변 후보 임베딩 미리 계산
answer_embs = encode_sentences(answer_texts, encoder)

def chatbot_answer(user_question, threshold=0.5):
    question_emb = encode_sentences([user_question], encoder)[0]
    sims = cosine_similarity([question_emb], answer_embs)[0]
    best_idx = sims.argmax()
    best_score = sims[best_idx]
    if best_score < threshold:
        return "죄송해요, 적절한 답변이 없어요."
    else:
        return answer_texts[best_idx]

# 테스트
if __name__ == "__main__":
    print(chatbot_answer("sklearn이 뭐야?"))
    print(chatbot_answer("환불 절차 알려주세요"))
    print(chatbot_answer("주문 취소 가능한가요?"))
    print(chatbot_answer("고객센터 연락처 알려줘"))
    print(chatbot_answer("이거 안돼요!"))
