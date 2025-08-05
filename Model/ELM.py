import pandas as pd
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.activations import gelu
import requests
import os
import pyarrow.dataset as ds
import sentencepiece as spm

# ==================== 파일 다운로드 함수 ====================
def download_file(url, save_path):
    if os.path.exists(save_path):
        print(f"✅ 이미 존재함: {save_path}")
        return
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"✅ 파일 저장됨: {save_path}")

import tensorflow as tf
print("✅ TF Version:", tf.__version__)
print("✅ GPU 디바이스 목록:", tf.config.list_physical_devices('GPU'))
print("✅ 사용 디바이스:", tf.test.gpu_device_name())


PARQUET_PATH = 'dataset.parquet'
MODEL_SAVE_PATH = 'sentence_encoder_model.keras'
SP_MODEL_PATH = "ko_bpe.model"  # SentencePiece 모델 경로

download_file(
    'https://huggingface.co/datasets/Yuchan5386/Chat2/resolve/main/dataset.parquet?download=true',
    PARQUET_PATH
)

download_file(
    'https://huggingface.co/datasets/Yuchan5386/Chat2/resolve/main/ko_bpe.model?download=true',
    SP_MODEL_PATH
)

sp = spm.SentencePieceProcessor()
sp.load(SP_MODEL_PATH)

# ==================== 하이퍼파라미터 ====================
vocab_size = sp.get_piece_size()
EMBED_DIM = 128
NUM_HEADS = 4
DENSE_UNITS = 256
MAX_SEQ_LEN = 128
OUTPUT_DIM = 128
EPOCHS = 1
BATCH_SIZE = 2048

def sp_tokenize(texts, max_len=MAX_SEQ_LEN):
    # ✅ 벡터화된 토크나이징 (빠르고 효율적)
    encoded = sp.encode(texts, out_type=int)  # list of list[int]
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        encoded, maxlen=max_len, padding='post', truncating='post'
    )
    return padded

# ==================== 데이터 파싱 ====================
def parse_conversation(text):
    try:
        obj = json.loads(text)
        convs = obj.get("conversation", {}).get("conversations", [])
    except Exception:
        return []

    qa_pairs = []
    for i in range(len(convs) - 1):
        if convs[i].get("from") == "human" and convs[i+1].get("from") == "gpt":
            q = convs[i].get("value", "").strip()
            a = convs[i+1].get("value", "").strip()
            if q and a:
                qa_pairs.append((q, a))
    return qa_pairs


from keras.saving import register_keras_serializable

import tensorflow as tf

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

def build_simple_encoder():
    inputs = tf.keras.layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32)
    x = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=EMBED_DIM, mask_zero=True)(inputs)
    
    attn = tf.keras.layers.MultiHeadAttention(num_heads=NUM_HEADS, key_dim=EMBED_DIM // NUM_HEADS)
    x = attn(x, x)
    x = tf.keras.layers.Dense(EMBED_DIM*2, activation='gelu')(x)
    x = tf.keras.layers.Dense(EMBED_DIM)(x)
    
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(EMBED_DIM, activation='gelu')(x)
    x = tf.keras.layers.Dense(OUTPUT_DIM)(x)
    outputs = L2NormLayer()(x)
    return tf.keras.Model(inputs, outputs, name="SimpleEncoder")

def build_simple_siamese_model():
    encoder = build_simple_encoder()
    input_q = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_question")
    input_a = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32, name="input_answer")
    emb_q = encoder(input_q)
    emb_a = encoder(input_a)
    outputs = layers.Concatenate(axis=0)([emb_q, emb_a])
    return Model(inputs=[input_q, input_a], outputs=outputs), encoder


def triplet_loss(y_true, y_pred, margin=0.2048):
    batch_size = tf.shape(y_pred)[0] // 2
    q_embs = y_pred[:batch_size]
    a_embs = y_pred[batch_size:]

    pos_sim = tf.reduce_sum(q_embs * a_embs, axis=1)
    neg_sim = tf.matmul(q_embs, a_embs, transpose_b=True)
    eye_mask = tf.eye(batch_size, dtype=tf.bool)
    neg_sim = tf.where(eye_mask, -1e9, neg_sim)
    hardest_neg_sim = tf.reduce_max(neg_sim, axis=1)

    loss = tf.maximum(0.0, margin + hardest_neg_sim - pos_sim)
    return tf.reduce_mean(loss)
    
def parquet_streaming_generator(parquet_path, batch_size=32):
    dataset = ds.dataset(parquet_path, format="parquet")
    scanner = ds.Scanner.from_dataset(dataset, batch_size=batch_size)
    
    questions = []
    answers = []
    
    for record_batch in scanner.to_batches():
        batch_df = record_batch.to_pandas()
        for idx, row in batch_df.iterrows():
            qa_pairs = parse_conversation(row.get("conversation", ""))
            for q, a in qa_pairs:
                questions.append(q)
                answers.append(a)

                # batch_size 개 쌓이면 yield
                if len(questions) == batch_size:
                    q_seq = sp_tokenize(questions, max_len=MAX_SEQ_LEN)
                    a_seq = sp_tokenize(answers, max_len=MAX_SEQ_LEN)
                    yield (q_seq, a_seq), np.zeros(batch_size, dtype=np.float32)
                    questions = []
                    answers = []
    
    # 마지막 남은 거 batch_size 안되도 yield (선택사항)
    if len(questions) > 0:
        q_seq = sp_tokenize(questions, max_len=MAX_SEQ_LEN)
        a_seq = sp_tokenize(answers, max_len=MAX_SEQ_LEN)
        yield (q_seq, a_seq), np.zeros(len(questions), dtype=np.float32)

def create_tf_dataset(parquet_path, batch_size):
    dataset = tf.data.Dataset.from_generator(
        lambda: parquet_streaming_generator(parquet_path, batch_size),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, MAX_SEQ_LEN), dtype=tf.int32),
                tf.TensorSpec(shape=(None, MAX_SEQ_LEN), dtype=tf.int32),
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        )
    )
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

def encode_sentences(sentences, encoder):
    seqs = sp_tokenize(sentences, max_len=MAX_SEQ_LEN)
    dataset = tf.data.Dataset.from_tensor_slices(seqs).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return encoder.predict(dataset, verbose=0)  

if __name__ == "__main__":
    siamese_model, encoder = build_simple_siamese_model()
    siamese_model.compile(optimizer='adam', loss=triplet_loss)
    print("✅ Siamese 모델 생성 완료")
    print(siamese_model.summary())

    total_qa_count = 266958
    steps_per_epoch = total_qa_count // BATCH_SIZE

    # ❗잘못된 generator 전달 고침
    train_dataset = create_tf_dataset(PARQUET_PATH, BATCH_SIZE)


    print("🚀 학습 시작...")
    siamese_model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        verbose=1
    )

    # ✅ 인코더 & 전체 모델 저장
    encoder.save(MODEL_SAVE_PATH)
    siamese_model.save("siamese_full_model.keras")
    print(f"✅ 모델 저장 완료: 인코더={MODEL_SAVE_PATH}, 전체=siamese_full_model")

    # ✅ 추론 테스트
    test_questions = ["배송은 언제 오나요?", "환불은 어떻게 하나요?", "주문 취소할 수 있나요?"]
    test_embeddings = encode_sentences(test_questions, encoder)
    print(f"✅ 테스트 문장 임베딩 생성 완료: {test_embeddings.shape}")

    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity([test_embeddings[0]], [test_embeddings[1]])
    print(f"임베딩[0] vs [1] 유사도: {sim[0][0]:.4f}")  