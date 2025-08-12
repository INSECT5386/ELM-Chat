import tensorflow as tf
from tensorflow.keras.models import load_model
import sentencepiece as spm
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras import layers
from tokenizers import Tokenizer

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
    

# 하이퍼파라미터
MAX_SEQ_LEN = 128
BATCH_SIZE = 1024
TK_MODEL_PATH = r'C:\Users\yuchan\Serve_Project\ko_bpe.json'
MODEL_SAVE_PATH = r'C:\Users\yuchan\Serve_Project\VeELM\sentence_encoder_model.keras'

tokenizer = Tokenizer.from_file(TK_MODEL_PATH)
def tk_tokenize(texts, max_len=MAX_SEQ_LEN):
    if isinstance(texts, str):
        texts = [texts]
    encoded = [tokenizer.encode(text).ids for text in texts]
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        encoded, maxlen=max_len, padding='post', truncating='post'
    )
    return padded


encoder = load_model(MODEL_SAVE_PATH, custom_objects={"L2NormLayer": L2NormLayer, "LearnableWeightedPooling": LearnableWeightedPooling})

def encode_sentences(sentences, encoder):
    seqs = tk_tokenize(sentences, max_len=MAX_SEQ_LEN)
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

a ="""

The request asks for a function to determine if a number is a "strong prime" based on three conditions, with constraints on allowed tools and time complexity. Let's break down the problem and devise a solution.

**Understanding the Requirements:**

1. **Input:** An integer.
2. **Output:** Boolean (True if the number is a strong prime, False otherwise).
3. **Strong Prime Conditions:**
a. Greater than 1.
b. Not a perfect square.
c. Not a multiple of the first 100 prime numbers.
4. **Constraints:**
a. Cannot use math libraries or built-in functions for checking perfect squares or divisibility.
b. Time complexity must be less than O(sqrt(n)).

**Condition Analysis and Algorithm Design:**

* **Condition 1: Greater than 1:** This is a simple check.

* **Condition 2: Not a perfect square:** We need to determine if a number `n` is a perfect square without using `sqrt` or `math.sqrt`. We can iterate from 1 upwards, squaring each number and comparing it to `n`. If `i*i == n`, then `n` is a perfect square. To keep the time complexity under control, we can stop the iteration when `i*i > n`.

* **Condition 3: Not a multiple of the first 100 prime numbers:** This requires us to know the first 100 prime numbers. We'll need a way to generate these. Once we have them, we can check if the input number is divisible by any of these primes.

**Generating the First 100 Primes:**

We can use the Sieve of Eratosthenes to efficiently generate prime numbers up to a certain limit. However, since we only need the first 100, a simpler approach might be to generate primes sequentially until we have 100 of them.

Algorithm for generating primes:

1. Start with a list of primes containing 2.
2. Initialize a number to check, `num = 3`.
3. While the number of primes in the list is less than 100:
a. Assume `num` is prime (`is_prime = True`).
b. Iterate through the existing primes in our list.
c. If `num` is divisible by any of the primes in the list, set `is_prime = False` and break the inner loop.
d. If `is_prime` is still `True` after checking all existing primes, add `num` to the list of primes.
e. Increment `num` by 2 (we only need to check odd numbers after 2).

**Checking Divisibility Without the Modulo Operator:**

The prompt explicitly restricts the use of math libraries and built-in functions. This likely includes the modulo operator (`%`). We need to perform divisibility checks without it.

Divisibility can be checked using repeated subtraction. To check if `a` is divisible by `b`, we can repeatedly subtract `b` from `a` until the result is less than `b`. If the final result is 0, then `a` is divisible by `b`. However, this can be inefficient for large numbers.

A more efficient approach without the modulo operator is to use integer division and multiplication. If `a // b * b == a`, then `a` is divisible by `b`.

**Putting it all together in the `is_strong_prime` function:**

```python
def is_strong_prime(n):
# Condition 1: Greater than 1
if n <= 1:
return False

# Condition 2: Not a perfect square
is_perfect_square = False
i = 1
while i * i <= n:
if i * i == n:
is_perfect_square = True
break
i += 1
if is_perfect_square:
return False

# Condition 3: Not a multiple of the first 100 prime numbers
first_100_primes = get_first_100_primes()
for prime in first_100_primes:
if n // prime * prime == n: # Check for divisibility without modulo
return False

return True

def get_first_100_primes():
primes = [2]
num = 3
while len(primes) < 100:
is_prime = True
for p in primes:
if num // p * p == num: # Check for divisibility
is_prime = False
break
if is_prime:
primes.append(num)
num += 2
return primes

```

**Time Complexity Analysis:**

* **`get_first_100_primes`:** Generating the first 100 primes involves checking divisibility up to the square root of the number being tested. The largest of the first 100 primes is relatively small (around 541). The complexity is roughly proportional to the sum of the square roots of the numbers up to the 100th prime, which is manageable.

* **`is_strong_prime`:**
* Perfect square check: The loop runs up to `sqrt(n)`, so the complexity is O(sqrt(n)).
* Divisibility by the first 100 primes: This involves iterating through a fixed list of 100 primes. The divisibility check using integer division and multiplication is O(1). Therefore, this part is O(100) which is O(1).

The overall time complexity of `is_strong_prime` is dominated by the perfect square check, which is O(sqrt(n)). This meets the requirement of being less than O(sqrt(n)). *Correction: The prompt says the time complexity should be *strictly* less than O(sqrt(n)).* This means we need to optimize further or rethink our approach.

**Optimization:**

The perfect square check is currently O(sqrt(n)). To improve this, we can leverage properties of perfect squares. Notice that the difference between consecutive perfect squares increases: 1, 4, 9, 16... differences are 3, 5, 7...

Consider the last digit of perfect squares: 0, 1, 4, 5, 6, 9. If the input number's last digit is not one of these, it cannot be a perfect square. This check is O(1).

However, this only rules out some non-perfect squares. To strictly meet the < O(sqrt(n)) time complexity, we might need to avoid iterating up to `sqrt(n)` for the perfect square check.

**Alternative Approach for Perfect Square Check:**

We can try a binary search approach. Search for an integer `i` such that `i * i == n`. The search space is from 1 to `n`. Binary search takes O(log n) time, which is less than O(sqrt(n)).

```python
def is_perfect_square_log_n(n):
if n < 0:
return False
if n == 0:
return True

low = 1
high = n

while low <= high:
mid = (low + high) // 2
mid_squared = mid * mid

if mid_squared == n:
return True
elif mid_squared < n:
low = mid + 1
else:
high = mid - 1
return False
```

Now, let's integrate this into the `is_strong_prime` function.

```python
def is_strong_prime(n):
# Condition 1: Greater than 1
if n <= 1:
return False

# Condition 2: Not a perfect square (using O(log n) approach)
if is_perfect_square_log_n(n):
return False

# Condition 3: Not a multiple of the first 100 prime numbers
first_100_primes = get_first_100_primes()
for prime in first_100_primes:
if n // prime * prime == n: # Check for divisibility without modulo
return False

return True
```

With the binary search for perfect square, the dominant time complexity is now O(log n), satisfying the constraint.

**Final Code Structure:**

```python
def get_first_100_primes():
primes = [2]
num = 3
while len(primes) < 100:
is_prime = True
for p in primes:
if num // p * p == num:
is_prime = False
break
if is_prime:
primes.append(num)
num += 2
return primes

def is_perfect_square_log_n(n):
if n < 0:
return False
if n == 0:
return True

low = 1
high = n

while low <= high:
mid = (low + high) // 2
mid_squared = mid * mid

if mid_squared == n:
return True
elif mid_squared < n:
low = mid + 1
else:
high = mid - 1
return False

def is_strong_prime(n):
if n <= 1:
return False

if is_perfect_square_log_n(n):
return False

first_100_primes = get_first_100_primes()
for prime in first_100_primes:
if n // prime * prime == n:
return False

return True

# Example usage
number = 11
is_strong = is_strong_prime(number)
print(f"{number} is a strong prime: {is_strong}")

number = 4 # Perfect square
is_strong = is_strong_prime(number)
print(f"{number} is a strong prime: {is_strong}")

number = 6 # Multiple of 2
is_strong = is_strong_prime(number)
print(f"{number} is a strong prime: {is_strong}")

# Test with a prime number not in the first 100
number = 101
is_strong = is_strong_prime(number)
print(f"{number} is a strong prime: {is_strong}") # Should be True
```

One important note: The prompt asks to check if the number is a *prime* number that satisfies the conditions. My current implementation checks if the number satisfies the three conditions. I need to ensure the input number itself is prime.

**Revising the `is_strong_prime` function to check for primality:**

```python
def is_prime(num):
if num <= 1:
return False
if num <= 3:
return True
if num % 2 == 0 or num % 3 == 0:
return False
i = 5
while i * i <= num:
if num % i == 0 or num % (i + 2) == 0:
return False
i += 6
return True

def is_strong_prime(n):
if not is_prime(n):
return False

if n <= 1:
return False # Redundant given is_prime check, but for clarity

if is_perfect_square_log_n(n):
return False

first_100_primes = get_first_100_primes()
for prime in first_100_primes:
if n // prime * prime == n:
return False

return True
```

The `is_prime` function uses a standard optimization by checking divisibility by 2 and 3 and then iterating with a step of 6. However, we cannot use the modulo operator here.

**Prime Check without Modulo:**

```python
def is_divisible(a, b):
return a // b * b == a

def is_prime_without_modulo(num):
if num <= 1:
return False
if num <= 3:
return True
if is_divisible(num, 2) or is_divisible(num, 3):
return False
i = 5
while i * i <= num:
if is_divisible(num, i) or is_divisible(num, i + 2):
return False
i += 6
return True

def is_strong_prime(n):
if not is_prime_without_modulo(n):
return False

if n <= 1:
return False

if is_perfect_square_log_n(n):
return False

first_100_primes = get_first_100_primes()
for prime in first_100_primes:
if n // prime * prime == n:
return False

return True
```

The `is_prime_without_modulo` function has a time complexity of roughly O(sqrt(n)), which might make the overall `is_strong_prime` complexity O(sqrt(n)). Let's reconsider the requirements.

The problem statement implies that the input number itself needs to be prime in addition to satisfying the other conditions.

**Corrected Interpretation:**

A strong prime is a *prime number* that satisfies the three conditions.

Algorithm:

1. Check if the input number `n` is prime (without using modulo).
2. If it is prime, then check the three conditions:
a. `n > 1` (already covered by primality check).
b. `n` is not a perfect square (using O(log n) method).
c. `n` is not a multiple of the first 100 primes (without modulo).

**Final Implementation:**

```python
def is_divisible(a, b):
return a // b * b == a

def is_prime_without_modulo(num):
if num <= 1:
return False
if num <= 3:
return True
if is_divisible(num, 2) or is_divisible(num, 3):
return False
i = 5
while i * i <= num:
if is_divisible(num, i) or is_divisible(num, i + 2):
return False
i += 6
return True

def get_first_100_primes():
primes = [2]
num = 3
while len(primes) < 100:
is_prime = True
for p in primes:
if is_divisible(num, p):
is_prime = False
break
if is_prime:
primes.append(num)
num += 2 if num > 2 else 1
return primes

def is_perfect_square_log_n(n):
if n < 0:
return False
if n == 0:
return True

low = 1
high = n
while low <= high:
mid = (low + high) // 2
mid_squared = mid * mid
if mid_squared == n:
return True
elif mid_squared < n:
low = mid + 1
else:
high = mid - 1
return False

def is_strong_prime(n):
if not is_prime_without_modulo(n):
return False

# Condition 1: > 1 (covered by is_prime)

# Condition 2: Not a perfect square
if is_perfect_square_log_n(n):
return False

# Condition 3: Not a multiple of the first 100 primes
first_100_primes = get_first_100_primes()
for prime in first_100_primes:
if is_divisible(n, prime):
return False

return True


```python
def is_divisible(a, b):
return a // b * b == a

def is_prime_without_modulo(num):
if num <= 1:
return False
if num <= 3:
return True
if is_divisible(num, 2) or is_divisible(num, 3):
return False
i = 5
while i * i <= num:
if is_divisible(num, i) or is_divisible(num, i + 2):
return False
i += 6
return True

def get_first_100_primes():
primes = [2]
num = 3
while len(primes) < 100:
is_prime = True
for p in primes:
if is_divisible(num, p):
is_prime = False
break
if is_prime:
primes.append(num)
num += 2
return primes

def is_perfect_square_log_n(n):
if n < 0:
return False
if n == 0:
return True

low = 1
high = n
while low <= high:
mid = (low + high) // 2
mid_squared = mid * mid
if mid_squared == n:
return True
elif mid_squared < n:
low = mid + 1
else:
high = mid - 1
return False

def is_strong_prime(n):
if not is_prime_without_modulo(n):
return False

if is_perfect_square_log_n(n):
return False

first_100_primes = get_first_100_primes()
for prime in first_100_primes:
if is_divisible(n, prime):
return False

return True
```
"""


# 답변 후보 (예시)
answer_texts = [
    code,
    "안녕하세요. 반가워요",
    "다음은 사이킷런을 사용하여 KNN을 구현하는 예시입니다. ```python import sklearn",
    "다음은 파이썬을 사용하여",
    a

]

q = """
입력된 숫자가 강력한 소수인지 아닌지를 확인하는 함수를 만듭니다.

강력한 소수는 다음 조건을 만족하는 소수입니다:
1. 1보다 큽니다.
2. 완벽한 정사각형이 아니다.
3. 처음 100개의 소수의 배수가 아니다.

참고: 완벽한 정사각형이나 소수의 배수를 확인하기 위해 수학 라이브러리나 내장 함수를 사용할 수 없습니다. 솔루션의 시간 복잡도는 n이 주어진 숫자인 O(sqrt(n))보다 작아야 합니다.
"""

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
    print(chatbot_answer("sklearn으로 KNN 만들어줘"))
    print(chatbot_answer(q))