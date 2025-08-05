import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sentencepiece as spm
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from huggingface_hub import hf_hub_download

# GPU 메모리 그로스 설정 (필수 아님, 선택사항)
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

# 경로 설정 (Huggingface hub에서 자동 다운로드)
hf_token = os.getenv("HF_TOKEN")
SP_MODEL_PATH = hf_hub_download(repo_id="Yuchan5386/ELM", filename="ko_bpe.model", repo_type="model", token=hf_token)
MODEL_PATH = hf_hub_download(repo_id="Yuchan5386/ELM", filename="sentence_encoder_model.keras", repo_type="model", token=hf_token)
PARQUET_PATH = hf_hub_download(repo_id="Yuchan5386/Chat2", filename="dataset.parquet", repo_type="dataset", token=hf_token)

MAX_SEQ_LEN = 128
BATCH_SIZE = 512  # 스트리밍용으로 줄임

# SentencePiece 로드
sp = spm.SentencePieceProcessor()
sp.load(SP_MODEL_PATH)

# 커스텀 L2Norm 레이어 정의 (모델 불러올 때 필요)
class L2NormLayer(tf.keras.layers.Layer):
    def __init__(self, axis=1, epsilon=1e-10, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=self.axis, epsilon=self.epsilon)
    def get_config(self):
        return {"axis": self.axis, "epsilon": self.epsilon, **super().get_config()}

# 인코더 모델 로드
encoder = load_model(MODEL_PATH, custom_objects={"L2NormLayer": L2NormLayer})
print("✅ 인코더 모델 로드 완료")

def sp_tokenize(texts):
    encoded = sp.encode(texts, out_type=int)
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        encoded, maxlen=MAX_SEQ_LEN, padding='post', truncating='post'
    )
    return padded

def parquet_streaming_generator(parquet_path, batch_size):
    dataset = ds.dataset(parquet_path, format="parquet")
    scanner = ds.Scanner.from_dataset(dataset, batch_size=batch_size)
    
    answers = []
    for record_batch in scanner.to_batches():
        batch_df = record_batch.to_pandas()
        for conv_json in batch_df['conversation']:
            obj = json.loads(conv_json)
            conv_list = obj.get("conversation", {}).get("conversations", [])
            for turn in conv_list:
                if turn.get('from') == 'gpt' and 'value' in turn:
                    answers.append(turn['value'])
                    if len(answers) == batch_size:
                        yield sp_tokenize(answers)
                        answers = []
    if answers:
        yield sp_tokenize(answers)

def encode_sentences_streaming(parquet_path):
    embeddings_list = []
    for a_seq in parquet_streaming_generator(parquet_path, BATCH_SIZE):
        dataset = tf.data.Dataset.from_tensor_slices(a_seq).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        embs = encoder.predict(dataset, verbose=0)
        embeddings_list.append(embs)
        print(f"스트리밍 배치 임베딩 완료, 누적 임베딩 수: {sum([e.shape[0] for e in embeddings_list])}")
    embeddings = np.vstack(embeddings_list)
    return embeddings.astype('float32')

if __name__ == "__main__":
    print("🧠 임베딩 스트리밍 계산 시작...")
    answer_embs = encode_sentences_streaming(PARQUET_PATH)
    print(f"✅ 임베딩 생성 완료: shape={answer_embs.shape}")

    print("💾 임베딩 압축 저장 시작...")
    np.savez_compressed('answer_embeddings_streaming.npz', embeddings=answer_embs)
    print("✅ 임베딩 저장 완료: answer_embeddings_streaming.npz")
