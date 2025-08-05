import sentencepiece as spm

# 1) 모델 로드
sp = spm.SentencePieceProcessor()
sp.load('ko_bpe.model')  # 경로 맞게 지정!

# 2) 토크나이징 예시
text = "안녕하세요"
tokens = sp.encode(text, out_type=str)  # subword 토큰 리스트로 출력
print("Tokens:", tokens)

# 3) 토큰 아이디 리스트
token_ids = sp.encode(text, out_type=int)
print("Token IDs:", token_ids)

# 4) 디코딩
decoded_text = sp.decode(token_ids)
print("Decoded:", decoded_text)
