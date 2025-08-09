import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer
# GPU 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU {len(gpus)}개 메모리 그로스 설정 완료")
    except Exception as e:
        print("⚠️ GPU 메모리 그로스 설정 실패:", e)
else:
    print("⚠️ GPU 디바이스 없음")

# 경로 설정
TK_MODEL_PATH = hf_hub_download(repo_id="Yuchan5386/Kode", filename="ko_bpe.json", repo_type="dataset")
MODEL_PATH = hf_hub_download(repo_id="Yuchan5386/VeELM", filename="sentence_encoder_model.keras", repo_type="model")
JSONL_PATH = hf_hub_download(repo_id="Yuchan5386/Kode", filename="data.jsonl", repo_type="dataset")

MAX_SEQ_LEN = 128
BATCH_SIZE = 32

# tokenizers 라이브러리 토크나이저 로드
tokenizer = Tokenizer.from_file(TK_MODEL_PATH)
vocab_size = tokenizer.get_vocab_size()

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

class MaskedGlobalAveragePooling1D(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs, mask=None):
        # inputs: (batch, seq_len, dim)
        # mask:   (batch, seq_len)  — True for real tokens, False for PAD
        if mask is None:
            return tf.reduce_mean(inputs, axis=1)
        mask = tf.cast(mask, inputs.dtype)               # (batch, seq_len)
        mask = tf.expand_dims(mask, axis=-1)             # (batch, seq_len, 1)
        summed = tf.reduce_sum(inputs * mask, axis=1)    # (batch, dim)
        counts = tf.reduce_sum(mask, axis=1)             # (batch, 1)
        return summed / (counts + 1e-9)                  # 0 나누기 방지

    def compute_mask(self, inputs, mask=None):
        # 이 레이어 뒤에는 더 이상 mask가 필요 없으니 None 반환
        return None

encoder = load_model(MODEL_PATH, custom_objects={"L2NormLayer": L2NormLayer, "MaskedGlobalAveragePooling1D": MaskedGlobalAveragePooling1D})

print("✅ 인코더 모델 로드 완료")

def tk_tokenize(texts):
    encoded = [tokenizer.encode(text).ids for text in texts]
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        encoded, maxlen=MAX_SEQ_LEN, padding='post', truncating='post'
    )
    return padded

def jsonl_streaming_generator(jsonl_path, batch_size):
    answers = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                convs = obj.get("conversations", [])
                for turn in convs:
                    if turn.get("from") == "gpt" and "value" in turn:
                        answers.append(turn["value"])
                        if len(answers) == batch_size:
                            yield tk_tokenize(answers)
                            answers = []
            except Exception:
                continue
    if answers:
        yield tk_tokenize(answers)

# 스트리밍 임베딩 인코딩
def encode_sentences_streaming(jsonl_path):
    embeddings_list = []
    for a_seq in jsonl_streaming_generator(jsonl_path, BATCH_SIZE):
        dataset = tf.data.Dataset.from_tensor_slices(a_seq).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        embs = encoder.predict(dataset, verbose=0)
        embeddings_list.append(embs)
        print(f"스트리밍 배치 임베딩 완료, 누적 임베딩 수: {sum([e.shape[0] for e in embeddings_list])}")
    embeddings = np.vstack(embeddings_list)
    return embeddings.astype('float32')

# 메인 실행
if __name__ == "__main__":
    print("🧠 임베딩 스트리밍 계산 시작...")
    answer_embs = encode_sentences_streaming(JSONL_PATH)
    print(f"✅ 임베딩 생성 완료: shape={answer_embs.shape}")

    print("💾 임베딩 압축 저장 시작...")
    np.savez_compressed('answer_embeddings_streaming.npz', embeddings=answer_embs)
    print("✅ 임베딩 저장 완료: answer_embeddings_streaming.npz")