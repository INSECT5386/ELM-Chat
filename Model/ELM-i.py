import tensorflow as tf
from tensorflow.keras.models import load_model
import sentencepiece as spm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras import layers

class L2NormLayer(layers.Layer):
    def __init__(self, axis=1, epsilon=1e-10, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=self.axis, epsilon=self.epsilon)
    def get_config(self):
        return {"axis": self.axis, "epsilon": self.epsilon, **super().get_config()}

# ================== Masked Global Average Pooling ==================
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
    
# 하이퍼파라미터
MAX_SEQ_LEN = 128
BATCH_SIZE = 1024
SP_MODEL_PATH = r'C:\Users\yuchan\Serve_Project\VeELM\ko_bpe.model'
MODEL_SAVE_PATH = r'C:\Users\yuchan\Serve_Project\VeELM\sentence_encoder_model.keras'

# SentencePiece 로드
sp = spm.SentencePieceProcessor()
sp.load(SP_MODEL_PATH)

def sp_tokenize(texts, max_len=MAX_SEQ_LEN):
    encoded = sp.encode(texts, out_type=int)
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        encoded, maxlen=max_len, padding='post', truncating='post'
    )
    return padded

encoder = load_model(MODEL_SAVE_PATH, custom_objects={"L2NormLayer": L2NormLayer, "MaskedGlobalAveragePooling1D": MaskedGlobalAveragePooling1D})

def encode_sentences(sentences, encoder):
    seqs = sp_tokenize(sentences, max_len=MAX_SEQ_LEN)
    dataset = tf.data.Dataset.from_tensor_slices(seqs).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return encoder.predict(dataset, verbose=0)

code = """
다음은 배열에서 인접한 두 요소의 최대 합을 구하는 Java 함수의 구현입니다:

```java
public class Main {
public static int maxAdjacentSum(int[] nums) {
int maxSum = 0;
for (int i = 0; i < nums.length - 1; i++) {
int sum = nums[i] + nums[i + 1];
if (sum > maxSum) {
maxSum = sum;
}
}
return maxSum;
}

public static void main(String[] args) {
int[] nums = {1, 2, 3, 4, 5};
System.out.println(maxAdjacentSum(nums)); // Output: 9
}
}
```

입력 배열에 음의 정수가 포함된 경우를 처리하려면 다음과 같이 구현을 수정할 수 있습니다:

```java
public class Main {
public static int maxAdjacentSum(int[] nums) {
int maxSum = Integer.MIN_VALUE;
for (int i = 0; i < nums.length - 1; i++) {
int sum = nums[i] + nums[i + 1];
if (sum > maxSum) {
maxSum = sum;
}
}
return maxSum;
}

public static void main(String[] args) {
int[] nums = {1, -2, 3, 4, 5};
System.out.println(maxAdjacentSum(nums)); // Output: 9
}
}
```

배열에서 인접하지 않은 두 요소의 최대 합을 구하려면 다음과 같이 수정된 버전의 카데인 알고리즘을 사용할 수 있습니다:

```java
public class Main {
public static int maxNonAdjacentSum(int[] nums) {
int inclusive = nums[0];
int exclusive = 0;
for (int i = 1; i < nums.length; i++) {
int temp = inclusive;
inclusive = Math.max(inclusive, exclusive + nums[i]);
exclusive = temp;
}
return Math.max(inclusive, exclusive);
}

public static void main(String[] args) {
int[] nums = {1, 2, 3, 4, 5};
System.out.println(maxNonAdjacentSum(nums)); // Output: 9
}
}
```

이 수정된 버전에서는 "포함" 변수는 현재 요소를 포함하는 인접하지 않은 요소의 최대 합을 저장하고, "제외" 변수는 현재 요소를 제외한 인접하지 않은 요소의 최대 합을 저장합니다. 루프는 배열을 반복하여 각 단계에서 "포함"과 "제외"의 값을 업데이트합니다. 마지막으로 "포함"과 "제외"의 최대값이 결과로 반환됩니다.
"""

# 답변 후보 (예시)
answer_texts = [
    code,
    "안녕하세요. 반가워요",

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
    print(chatbot_answer("양수 배열을 받아 배열에 있는 인접한 두 요소의 최대 합을 반환하는 Java 함수를 구현합니다. 이 함수의 시간 복잡도는 O(n)이어야 하며, 여기서 n은 입력 배열의 길이입니다. 또한 이 함수는 내장된 정렬 또는 검색 함수를 사용해서는 안 됩니다.또한 이 함수는 입력 배열에 음의 정수가 포함된 경우도 처리해야 합니다. 이 경우 함수는 양수와 음수를 모두 포함하여 인접한 두 요소의 최대 합을 찾아야 합니다.난이도를 높이기 위해 함수를 수정하여 배열에서 인접하지 않은 두 요소의 최대 합을 반환하도록 합니다. 즉, 함수는 배열에서 두 요소가 서로 인접하지 않은 두 요소의 최대 합을 찾아야 합니다.예를 들어 입력 배열이 [1, 2, 3, 4, 5]인 경우, 두 요소가 인접하지 않은 두 요소의 최대 합은 4 + 5 = 9이므로 함수는 9를 반환해야 합니다."))
    print(chatbot_answer("안녕하세요."))

