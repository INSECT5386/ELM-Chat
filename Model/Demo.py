from flask import Flask, request, Response
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import sentencepiece as spm
import numpy as np
import json
import time
import re
import random
import os
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from huggingface_hub import hf_hub_download

app = Flask(__name__, static_folder="static")

# === Custom layers ===
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
        if mask is None:
            return tf.reduce_mean(inputs, axis=1)
        mask = tf.cast(mask, inputs.dtype)
        mask = tf.expand_dims(mask, axis=-1)
        summed = tf.reduce_sum(inputs * mask, axis=1)
        counts = tf.reduce_sum(mask, axis=1)
        return summed / (counts + 1e-9)
    def compute_mask(self, inputs, mask=None):
        return None

# === 환경 변수 및 토큰 ===
os.environ["HF_HOME"] = "/tmp/hf_cache"
hf_token = os.getenv("HF_TOKEN")

# === 상수 ===
MAX_SEQ_LEN = 128
BATCH_SIZE = 1024
SIM_THRESHOLD = 0.5  # 필요에 따라 조정 가능

# === 모델, 토크나이저, 데이터 다운로드 및 로드 ===
SP_MODEL_PATH = hf_hub_download(repo_id="Yuchan5386/VeELM", filename="ko_bpe.model", repo_type="model", token=hf_token)
MODEL_PATH = hf_hub_download(repo_id="Yuchan5386/VeELM", filename="sentence_encoder_model.keras", repo_type="model", token=hf_token)
JSONL_PATH = hf_hub_download(repo_id="Yuchan5386/Kode", filename="data.jsonl", repo_type="dataset", token=hf_token)
EMBEDDINGS_PATH = hf_hub_download(repo_id="Yuchan5386/VeELM", filename="answer_embeddings_streaming.npz", repo_type="model", token=hf_token)
INDEX_PATH = hf_hub_download(repo_id="Yuchan5386/VeELM", filename="answer_faiss_index.index", repo_type="model", token=hf_token)

# === SentencePiece 로드 ===
sp = spm.SentencePieceProcessor()
sp.load(SP_MODEL_PATH)

# === 인코더 로드 ===
encoder = load_model(MODEL_PATH, custom_objects={"L2NormLayer": L2NormLayer, "MaskedGlobalAveragePooling1D": MaskedGlobalAveragePooling1D})

# === 답변 텍스트 로드 ===
def load_answers_from_jsonl(jsonl_path):
    answers = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                convs = obj.get("conversations", [])
                for turn in convs:
                    if turn.get("from") == "gpt" and "value" in turn:
                        answers.append(turn["value"])
            except json.JSONDecodeError:
                continue
    return answers

answer_texts = load_answers_from_jsonl(JSONL_PATH)

# === 임베딩 로드 ===
embedding_npz = np.load(EMBEDDINGS_PATH)
answer_embs = embedding_npz['embeddings'].astype('float32')

# === faiss 인덱스 초기화 및 로드/생성 ===
dim = answer_embs.shape[1]
if os.path.isfile(INDEX_PATH):
    faiss_index = faiss.read_index(INDEX_PATH)
else:
    faiss_index = faiss.IndexFlatIP(dim)      # FAISS 인덱스를 faiss_index 변수에 담습니다.
    faiss_index.add(answer_embs)
    faiss.write_index(faiss_index, INDEX_PATH)


# === 토크나이징 & 패딩 함수 ===
def sp_tokenize(texts):
    encoded = sp.encode(texts, out_type=int)
    if isinstance(encoded[0], list):
        padded = tf.keras.preprocessing.sequence.pad_sequences(encoded, maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
    else:
        padded = tf.keras.preprocessing.sequence.pad_sequences([encoded], maxlen=MAX_SEQ_LEN, padding='post', truncating='post')
    return padded

# === 임베딩 생성 ===
def encode_sentences(texts):
    seqs = sp_tokenize(texts)
    dataset = tf.data.Dataset.from_tensor_slices(seqs).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return encoder.predict(dataset, verbose=0).astype('float32')

def remove_invalid_unicode(text):
    return re.sub(r'[\ud800-\udfff]', '', text)

def synonym_replace(text):
    replacements = {
        "좋아요": ["괜찮아요", "나쁘지 않아요", "좋지요"],
        "도와드릴게요": ["도와줄게요", "함께 해볼까요?", "내가 도와줄 수 있어요!"],
        "문제없어요": ["걱정 마세요", "아무 문제 없어요", "전혀 문제 없어요"],
        "할 수 있어요": ["가능해요", "할 수 있지요", "도전해봐요"],
        "알겠습니다": ["오키!", "확인했어요", "그렇게 할게요!"],
        "정확합니다": ["딱 맞아요!", "정확히 그거예요!", "맞았어요!"],
        "감사합니다": ["고마워요", "땡큐!", "감사해요~"],
        "죄송합니다": ["미안해요", "실례했어요", "어머 죄송해요!"],
        "안녕하세요": ["안녕!", "하이하이~", "안뇽~"],
    }
    for word, candidates in replacements.items():
        if word in text:
            text = text.replace(word, random.choice(candidates))
    return text

def shuffle_sentences(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) > 1:
        random.shuffle(sentences)
    return ' '.join(sentences)

def casual_tone(text):
    replacements = {
        "입니다": "예요", "할 수 있습니다": "할 수 있어요",
        "도와드리겠습니다": "도와줄게요", "감사합니다": "고마워요",
        "알겠습니다": "오케이~", "문제 없습니다": "문제 없어요~",
        "죄송합니다": "미안해요", "확인했습니다": "봤어요~",
        "잘 부탁드립니다": "잘 부탁해요~", "좋습니다": "좋아용", "괜찮습니다": "괜찮아용"
    }
    for formal, casual in replacements.items():
        text = text.replace(formal, casual)
    return text

def drop_redundant(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    if len(sentences) > 1:
        keep = random.sample(sentences, k=max(1, len(sentences)//2))
        return ' '.join(keep)
    return text

def reverse_phrase(text):
    sentences = re.split(r'(?<=[.!?]) +', text)
    return ' '.join(sentences[::-1])

def random_cut(text):
    words = text.split()
    if len(words) > 5:
        cut_point = random.randint(5, len(words))
        return ' '.join(words[:cut_point])
    return text

def insert_random_prefix(text):
    prefixes = [
        "사실 말하자면,", "개인적으로는,", "아마도 말이지,", "기억이 맞다면,",
        "요즘 생각나는 건데,", "이건 내 생각일 뿐인데,", "그니까,", "솔직히 말해서,",
        "아ㅋㅋ 근데,", "이건 진짜인데,", "진짜루다가,", "나만 이렇게 생각해?", "헿,"
    ]
    return f"{random.choice(prefixes)} {text}"

def add_suffix(text):
    suffixes = [
        " 맞죠?", " 그렇지 않아요?", " 그런 거 같아요!", " 흐음... 이상하죠?", " ㅎㅎ",
        " 아무튼 그렇습니다.", " 느낌 오죠?", " 뭐 그런 거예요~", " 이거 완전 꿀팁!", " 아시겠죠?",
        " 진심이에요!", " 요즘 다 이래요~", " 그냥 그렇다구요~"
    ]
    return text + random.choice(suffixes)

def add_emoji_or_slang(text):
    suffixes = [
        " 🤔", " 😆", " 😎", " 🫠", " ㅋㅋㅋ", " ㄹㅇ", " ㄴㅇㄱ", " 👀", " 😅", " 🤯", " 😂", " 🤫"
    ]
    return text + random.choice(suffixes)

def mix_with_similar_response(query_emb, base_response, top_k=3):
    D, I = faiss_index.search(query_emb, top_k)
    if len(I[0]) < 2:
        return base_response
    z_idx = I[0][1]
    z_text = answer_texts[z_idx]
    x_sentences = re.split(r'(?<=[.!?]) +', base_response)
    z_sentences = re.split(r'(?<=[.!?]) +', z_text)
    if not z_sentences:
        return base_response
    n = max(1, int(len(z_sentences) * 0.4))
    selected = random.sample(z_sentences, min(n, len(z_sentences)))
    insert_point = random.randint(0, len(x_sentences))
    new_sentences = x_sentences[:insert_point] + selected + x_sentences[insert_point:]
    return ' '.join(new_sentences)

MATH_PATTERNS = [
    r'\d+[\+\-\*\/]\d+',
    r'\([^\)]+\)\([^\)]+\)',
    r'[a-zA-Z0-9]+\^[a-zA-Z0-9]+',
    r'\d+\/\d+',
    r'sqrt\(|\√',
]

def contains_math(text):
    return any(re.search(pattern, text) for pattern in MATH_PATTERNS)

def solve_math_expression(expr: str) -> str:
    try:
        expr = expr.replace(' ', '').replace('×', '*').replace('÷', '/')
        expr = re.sub(r'\)\(', ')*(', expr)

        result = None
        steps = []

        simplified = expr
        numeric_matches = re.findall(r'\((\d+[\+\-\*\/]\d+)\)', expr)
        for match in numeric_matches:
            try:
                val = eval(match)
                simplified = simplified.replace(f"({match})", str(val))
                steps.append(f"먼저 괄호 안 계산: {match} = {val}")
            except:
                pass

        try:
            import sympy as sp
            expanded = sp.expand(simplified)
            original = sp.sympify(simplified)
            if str(expanded) != str(original):
                steps.append(f"식을 전개해 볼게요: {original} → {expanded}")
            result = str(expanded)
        except:
            try:
                result = str(eval(simplified))
                steps.append(f"계산 결과: {simplified} = {result}")
            except:
                result = None

        if result:
            explanation = " ".join(steps)
            if not explanation:
                explanation = "이 식을 계산해 볼게요."
            return f"{explanation} 최종 결과는 {result}이에요! ✨"
        else:
            return None
    except Exception:
        return None

FILTERED_SENTENCES = [
    "sklearn.preprocessing에서 StandardScaler를 가져옵니다.sklearn.decomposition에서 PCA를 가져옵니다.파이프라인 = 파이프라인(단계=[    ('스케일러', StandardScaler()),    ('pca', PCA())])",
    "단어와 각 단어의 해당 횟수가 포함된 사전을 설정하는 함수를 구성합니다.",
    """def create_sentence(words):    sentence = ""    missing_words = []    for word in words:        if word.endswith("."):            sentence += word        else:            sentence += word + " "    if len(missing_words) > 0:        print("다음 단어가 누락되었습니다: ")        for word in missing_words:            missing_word = input(f"{word}를 입력하세요: ")            sentence = sentence.replace(f"{word} ", f"{missing_word} ")    문장을 반환합니다.이 함수를 사용하는 방법은 다음과 같습니다:```pythonwords = ["이것", "이것", ... ]"""
]

def filter_response(text):
    for f_text in FILTERED_SENTENCES:
        if f_text in text:
            return ""
    return text

QUERY_EXPANSION_DICT = {
    "tensorflow": "import tensorflow as tf",
    "keras": "import tensorflow.keras as keras",
    "numpy": "import numpy as np",
    "pandas": "import pandas as pd",
    "sklearn": "import sklearn"
}

def expand_query(text):
    return QUERY_EXPANSION_DICT.get(text.strip().lower(), text)

def generate_response(user_input, top_k=3, similarity_threshold=SIM_THRESHOLD):
    user_input = expand_query(user_input)

    if contains_math(user_input):
        math_response = solve_math_expression(user_input)
        if math_response:
            if random.random() < 0.6:
                math_response = casual_tone(math_response)
            if random.random() < 0.5:
                math_response = add_suffix(math_response)
            if random.random() < 0.5:
                math_response = add_emoji_or_slang(math_response)
            return math_response

    AUG = {
        synonym_replace: 0.6,
        shuffle_sentences: 0.4,
        casual_tone: 0.5,
        drop_redundant: 0.3,
        reverse_phrase: 0.2,
        random_cut: 0.4,
        insert_random_prefix: 0.3,
    }
    augmented = user_input
    for f, p in AUG.items():
        if random.random() < p:
            augmented = f(augmented)

    query_emb = encode_sentences([augmented])
    D, I = faiss_index.search(query_emb, top_k)  # index → faiss_index 로 변경
    sims = D[0]
    best_idx = I[0][0]
    best_score = sims[0]

    if best_score < similarity_threshold:
        return random.choice([
            "음... 이건 그냥 넘길게요!", "그건 잘 모르겠어요. 😅",
            "이건 내 지식 밖인 것 같아요.", "패스! 다른 질문 가보자고~"
        ])

    best_response = answer_texts[best_idx]

    if random.random() < 0.5:
        best_response = mix_with_similar_response(query_emb, best_response)

    if random.random() < 0.5:
        best_response = casual_tone(best_response)

    if random.random() < 0.4:
        best_response = add_emoji_or_slang(best_response)

    return filter_response(best_response)


@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/chat', methods=['GET', 'POST'])
def chat_api():
    user_msg = request.json.get("message", "").strip() if request.method == "POST" else request.args.get("message", "").strip()

    if not user_msg:
        def err(): yield 'data: {"error":"메시지를 입력해 주세요."}\n\n'
        return Response(err(), mimetype='text/event-stream')

    def gen():
        try:
            resp = generate_response(user_msg)
            for ch in resp:
                safe = json.dumps(ch)[1:-1]
                yield f'data: {{"char":"{safe}"}}\n\n'
                time.sleep(0.004096)
            yield 'data: {"done":true}\n\n'
        except Exception as e:
            yield f'data: {{"error":"{str(e)}"}}\n\n'

    return Response(gen(), mimetype='text/event-stream')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
