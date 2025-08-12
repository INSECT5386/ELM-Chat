import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tokenizers import Tokenizer
import requests
import os
from tqdm import tqdm

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

print("✅ TF Version:", tf.__version__)
print("✅ GPU 디바이스 목록:", tf.config.list_physical_devices('GPU'))
print("✅ 사용 디바이스:", tf.test.gpu_device_name())

# 파일 경로 및 하이퍼파라미터
JSONL_PATH = 'data.jsonl'  # JSONL 파일명 확인 필수
MODEL_SAVE_PATH = 'sentence_encoder_model.keras'
TK_MODEL_PATH = "ko_bpe.json"  # tokenizers json 모델 파일

download_file(
    'https://huggingface.co/datasets/Yuchan5386/Kode/resolve/main/data.jsonl?download=true',
    JSONL_PATH
)
download_file(
    'https://huggingface.co/datasets/Yuchan5386/Kode/resolve/main/ko_bpe.json?download=true',  # json 파일 링크로 바꿔야 함
    TK_MODEL_PATH
)

# tokenizers 라이브러리 토크나이저 로드
tokenizer = Tokenizer.from_file(TK_MODEL_PATH)
vocab_size = tokenizer.get_vocab_size()

# 하이퍼파라미터
EMBED_DIM = 160
NUM_HEADS = 4
MAX_SEQ_LEN = 128
OUTPUT_DIM = 160
EPOCHS = 1
BATCH_SIZE = 128
TOTAL_QA_COUNT = 2682172

def tk_tokenize(texts, max_len=MAX_SEQ_LEN):
    encoded = [tokenizer.encode(text).ids for text in texts]
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        encoded, maxlen=max_len, padding='post', truncating='post'
    )
    return padded

# JSONL에서 QA쌍 추출
def parse_conversation(text):
    try:
        obj = json.loads(text)
        convs = obj.get("conversations", [])
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

# L2 정규화 레이어
class L2NormLayer(layers.Layer):
    def __init__(self, axis=1, epsilon=1e-10, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon

    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=self.axis, epsilon=self.epsilon)

    def get_config(self):
        return {"axis": self.axis, "epsilon": self.epsilon, **super().get_config()}

class LearnableWeightedPooling(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dense = None  # 초기엔 None으로 선언해둠

    def build(self, input_shape):
        # input_shape: (batch_size, seq_len, embed_dim)
        self.dense = layers.Dense(1, use_bias=False)
        self.dense.build(input_shape)  # Dense 레이어 build 호출해서 가중치 생성
        self.built = True  # build 완료 플래그 설정

    def call(self, inputs, mask=None):
        scores = self.dense(inputs)  # (batch, seq_len, 1)

        if mask is not None:
            mask = tf.cast(mask, scores.dtype)
            minus_inf = -1e9
            scores = scores + (1 - mask[..., tf.newaxis]) * minus_inf

        weights = tf.nn.softmax(scores, axis=1)  # (batch, seq_len, 1)
        weighted_sum = tf.reduce_sum(inputs * weights, axis=1)  # (batch, embed_dim)
        return weighted_sum

def transformer_encoder_block(x, num_heads=NUM_HEADS, ff_dim=EMBED_DIM*4):
    attn_output = layers.MultiHeadAttention(num_heads, EMBED_DIM//num_heads)(x, x)
    attn_output = layers.Dropout(0.1)(attn_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    ffn_output = layers.Dense(ff_dim, activation='gelu')(out1)
    ffn_output = layers.Dense(EMBED_DIM)(ffn_output)
    ffn_output = layers.Dropout(0.1)(ffn_output)
    return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)


def build_simple_encoder():
    inputs = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32)
    embedding = layers.Embedding(vocab_size, EMBED_DIM, mask_zero=True)(inputs)

    x = layers.LayerNormalization()(embedding)

    x = transformer_encoder_block(x)
    x = transformer_encoder_block(x)

    mask = embedding._keras_mask

    # 여기서 MaskedGlobalAveragePooling1D 대신 LearnableWeightedPooling 사용
    x = LearnableWeightedPooling()(x, mask=mask)

    x = layers.Dense(OUTPUT_DIM)(x)
    outputs = L2NormLayer()(x)
    return Model(inputs, outputs)


def build_simple_siamese_model():
    encoder = build_simple_encoder()
    input_q = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32)
    input_a = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32)

    emb_q = encoder(input_q)
    emb_a = encoder(input_a)

    return Model(inputs=[input_q, input_a], outputs=[emb_q, emb_a]), encoder

def jsonl_streaming_generator(jsonl_path, batch_size=BATCH_SIZE):
    questions, answers = [], []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            qa_pairs = parse_conversation(line)
            for q, a in qa_pairs:
                questions.append(q)
                answers.append(a)
                if len(questions) == batch_size:
                    q_seq = tk_tokenize(questions)
                    a_seq = tk_tokenize(answers)
                    yield (q_seq, a_seq), np.zeros(batch_size, dtype=np.float32)
                    questions.clear()
                    answers.clear()

    if questions:
        q_seq = tk_tokenize(questions)
        a_seq = tk_tokenize(answers)
        yield (q_seq, a_seq), np.zeros(len(questions), dtype=np.float32)

def create_tf_dataset_from_jsonl(jsonl_path, batch_size=BATCH_SIZE):
    dataset = tf.data.Dataset.from_generator(
        lambda: jsonl_streaming_generator(jsonl_path, batch_size),
        output_signature=(
            (
                tf.TensorSpec(shape=(None, MAX_SEQ_LEN), dtype=tf.int32),
                tf.TensorSpec(shape=(None, MAX_SEQ_LEN), dtype=tf.int32)
            ),
            tf.TensorSpec(shape=(None,), dtype=tf.float32)
        )
    )
    return dataset.prefetch(tf.data.AUTOTUNE)

def encode_sentences(sentences, encoder):
    seqs = tk_tokenize(sentences)
    dataset = tf.data.Dataset.from_tensor_slices(seqs).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return encoder.predict(dataset, verbose=0)

@tf.function
def nt_xent_loss(emb_q, emb_a, temperature=0.05):
    emb_q = tf.math.l2_normalize(emb_q, axis=1)
    emb_a = tf.math.l2_normalize(emb_a, axis=1)

    batch_size = tf.shape(emb_q)[0]
    logits = tf.matmul(emb_q, emb_a, transpose_b=True)
    logits /= temperature

    labels = tf.range(batch_size)
    loss_q = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    loss_a = tf.keras.losses.sparse_categorical_crossentropy(labels, tf.transpose(logits), from_logits=True)
    loss = (tf.reduce_mean(loss_q) + tf.reduce_mean(loss_a)) / 2
    return loss

@tf.function
def train_step(model, optimizer, batch_data):
    (q_seq, a_seq), _ = batch_data
    with tf.GradientTape() as tape:
        emb_q, emb_a = model([q_seq, a_seq], training=True)
        loss = nt_xent_loss(emb_q, emb_a)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


if __name__ == "__main__":
    siamese_model, encoder = build_simple_siamese_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    train_dataset = create_tf_dataset_from_jsonl(JSONL_PATH, BATCH_SIZE).repeat()

    steps_per_epoch = TOTAL_QA_COUNT // BATCH_SIZE

    print("✅ Siamese 모델 생성 완료")
    print(siamese_model.summary())

    print("🚀 학습 시작...")

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        pbar = tqdm(total=steps_per_epoch, desc="Training", ncols=100)

        for step, batch_data in enumerate(train_dataset):
            loss = train_step(siamese_model, optimizer, batch_data)
            pbar.set_postfix({"loss": f"{loss.numpy():.4f}"})
            pbar.update(1)

            if step + 1 >= steps_per_epoch:
                break

        pbar.close()

    print("✅ 학습 완료, 모델 저장 중...")
    encoder.save(MODEL_SAVE_PATH)
    siamese_model.save("siamese_full_model.keras")
    print(f"✅ 모델 저장 완료: {MODEL_SAVE_PATH}, siamese_full_model.keras")

    test_questions = ["배송은 언제 오나요?", "환불은 어떻게 하나요?", "주문 취소할 수 있나요?"]
    test_embeddings = encode_sentences(test_questions, encoder)
    print(f"✅ 테스트 문장 임베딩 생성 완료: {test_embeddings.shape}")

    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity([test_embeddings[0]], [test_embeddings[1]])
    print(f"임베딩[0] vs [1] 유사도: {sim[0][0]:.4f}") 

