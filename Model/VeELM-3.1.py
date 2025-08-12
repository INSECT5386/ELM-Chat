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
    'https://huggingface.co/datasets/Yuchan5386/Kode/resolve/main/ko_bpe.json?download=true',
    TK_MODEL_PATH
)

# tokenizers 라이브러리 토크나이저 로드
tokenizer = Tokenizer.from_file(TK_MODEL_PATH)
vocab_size = tokenizer.get_vocab_size()

# 하이퍼파라미터
EMBED_DIM = 160
NUM_HEADS = 4
MAX_SEQ_LEN = 192
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

# MultiHeadWeightedPooling 레이어 (learnable head weighting 포함)
class MultiHeadWeightedPooling(layers.Layer):
    def __init__(self, num_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads

    def build(self, input_shape):
        self.token_score_dense = layers.Dense(
            self.num_heads, use_bias=False,
            kernel_initializer='glorot_uniform'
        )
        self.head_weights = self.add_weight(
            name="head_weights",
            shape=(self.num_heads,),
            initializer=tf.keras.initializers.Ones(),
            trainable=True
        )
        super().build(input_shape)

    def call(self, inputs, mask=None):
        scores = self.token_score_dense(inputs)
        scores = scores - tf.reduce_max(scores, axis=1, keepdims=True)

        if mask is not None:
            mask = tf.cast(mask, scores.dtype)
            scores += (1 - mask[..., tf.newaxis]) * -1e9

        weights = tf.nn.softmax(scores, axis=1)
        head_outputs = tf.einsum("bse,bsn->bne", inputs, weights)
        head_weighted = head_outputs * tf.reshape(self.head_weights, (1, self.num_heads, 1))
        pooled = tf.reduce_sum(head_weighted, axis=1)
        return pooled

    def compute_mask(self, inputs, mask=None):
        return None  # Masking 끝

class GLALayer(layers.Layer):
    def __init__(self, dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0
        self.head_dim = dim // num_heads
        
        # 초기화 시점엔 None으로 둠
        self.qkv = None
        self.out_proj = None

    def build(self, input_shape):
        # input_shape: (batch, seq_len, embed_dim)
        self.qkv = layers.Dense(self.dim * 3)
        self.out_proj = layers.Dense(self.dim)
        super().build(input_shape)

    def call(self, inputs, mask=None):
        B, T = tf.shape(inputs)[0], tf.shape(inputs)[1]
        qkv = self.qkv(inputs)
        q, k, v = tf.split(qkv, 3, axis=-1)

        def split_heads(x):
            x = tf.reshape(x, (B, T, self.num_heads, self.head_dim))
            return tf.transpose(x, [0, 2, 1, 3])  # [B, H, T, hd]

        q = split_heads(q)
        k = split_heads(k)
        v = split_heads(v)

        k = tf.nn.softmax(k, axis=-1)

        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            mask = tf.reshape(mask, (B, 1, T, 1))
            k += (1 - mask) * -1e9
        k = tf.nn.softmax(k, axis=-1)


        context = tf.matmul(k, v, transpose_a=True)  # [B,H,hd,hd]
        out = tf.matmul(q, context)  # [B,H,T,hd]
        out = tf.transpose(out, [0, 2, 1, 3])  # [B,T,H,hd]
        out = tf.reshape(out, (B, T, self.dim))  # [B,T,D]
        return self.out_proj(out)

    def compute_mask(self, inputs, mask=None):
        return mask

def transformer_encoder_block(x, mask=None, num_heads=NUM_HEADS, ff_dim=EMBED_DIM*4):
    attn_output = GLALayer(dim=EMBED_DIM, num_heads=num_heads)(x, mask=mask)
    attn_output = layers.Dropout(0.1)(attn_output)
    out1 = layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

    ffn_output = layers.Dense(ff_dim, activation='gelu')(out1)
    ffn_output = layers.Dense(EMBED_DIM)(ffn_output)
    ffn_output = layers.Dropout(0.1)(ffn_output)
    return layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)


def build_simple_encoder():
    inputs = layers.Input(shape=(MAX_SEQ_LEN,), dtype=tf.int32)
    embedding_layer = layers.Embedding(vocab_size, EMBED_DIM, mask_zero=True)
    embedding = embedding_layer(inputs)

    x = layers.LayerNormalization()(embedding)

    mask = embedding_layer.compute_mask(inputs)

    x = transformer_encoder_block(x, mask=mask)
    x = transformer_encoder_block(x, mask=mask)

    x = MultiHeadWeightedPooling()(x, mask=mask)

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
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0)
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
