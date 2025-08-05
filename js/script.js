// 외부 라이브러리 동적 로드
(function loadLibraries() {
  // 1. marked.js 로드
  const markedScript = document.createElement('script');
  markedScript.src = 'https://cdn.jsdelivr.net/npm/marked/marked.min.js';
  markedScript.onload = () => {
    window.marked = marked; // 전역에서 사용 가능하게
    loadHighlightJS();
  };
  document.head.appendChild(markedScript);

  // 2. highlight.js 로드
  function loadHighlightJS() {
    const hljsScript = document.createElement('script');
    hljsScript.src = 'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.8.0/build/highlight.min.js';
    hljsScript.onload = () => {
      window.hljs = hljs;
      // hljs 초기화
      document.addEventListener('DOMContentLoaded', () => {
        hljs.configure({ ignoreIllegals: true });
      });
    };
    document.head.appendChild(hljsScript);
  }
})();

  // 상태 관리
  const state = {
    rooms: {},
    currentRoom: null
  };

  // DOM 요소
  const $ = (sel) => document.querySelector(sel);
  const roomListUl = $('#room-list ul');
  const inputText = $('#input-text');
  const sendBtn = $('#send-btn');
  const chatContainer = $('#chat-container');

  // 초기화
  function init() {
    createRoom('기본 채팅방');
    switchRoom('기본 채팅방');
    autoResize();

    // 테스트 코드 포함 메시지 추가 (하이라이팅 확인용)
    const sample = `
\`\`\`python
def greet(name):
    print(f"안녕, {name}!")
\`\`\`
`;
    addMessage(sample, false);
  }

  // 채팅방 생성
  function createRoom(name) {
    if (state.rooms[name]) {
      alert('이미 존재하는 방입니다!');
      return false;
    }
    state.rooms[name] = [];
    renderRoomList();
    return true;
  }

  // 채팅방 전환
  function switchRoom(name) {
    if (!state.rooms[name]) return;
    state.currentRoom = name;
    renderRoomList();
    renderMessages();
  }

  // 채팅방 목록 렌더링
  function renderRoomList() {
    roomListUl.innerHTML = '';
    Object.keys(state.rooms).forEach(roomName => {
      const li = document.createElement('li');
      li.textContent = roomName;
      if (roomName === state.currentRoom) li.classList.add('active');
      li.addEventListener('click', () => switchRoom(roomName));
      roomListUl.appendChild(li);
    });
  }

  // 메시지 렌더링
  function renderMessages() {
    chatContainer.innerHTML = '';
    const messages = state.rooms[state.currentRoom] || [];
    messages.forEach(msg => addMessageToUI(msg.text, msg.fromUser, false));
  }

  // 메시지 추가 (UI + 저장)
  function addMessage(text, fromUser, save = true) {
    addMessageToUI(text, fromUser, true);
    if (save) {
      state.rooms[state.currentRoom].push({ text, fromUser });
    }
  }

  // UI에 메시지 추가
  function addMessageToUI(text, fromUser, animate = true) {
    const div = document.createElement('div');
    div.classList.add('message', fromUser ? 'user' : 'bot');
    if (!animate) div.style.animation = 'none';
    div.innerHTML = marked.parse(text);

    div.querySelectorAll('pre code').forEach(block => {
      block.classList.add('hljs');            // hljs 강제 추가
      hljs.highlightElement(block);           // 문법 강조
      makeCopyButton(block);                  // 복사 버튼
    });

    chatContainer.appendChild(div);
    scrollToBottom();
  }

  // 복사 버튼 생성
  function makeCopyButton(codeBlock) {
    // 이미 복사 버튼 있으면 리턴
    if (codeBlock.parentNode.querySelector('.copy-btn')) return;

    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = '복사';
    btn.onclick = () => {
      navigator.clipboard.writeText(codeBlock.innerText)
        .then(() => {
          btn.textContent = '복사됨!';
          setTimeout(() => btn.textContent = '복사', 1500);
        })
        .catch(() => {
          btn.textContent = '실패';
          setTimeout(() => btn.textContent = '복사', 1500);
        });
    };
    codeBlock.parentNode.style.position = 'relative';
    codeBlock.parentNode.appendChild(btn);
  }

  // 자동 스크롤
  function scrollToBottom() {
    chatContainer.scrollTop = chatContainer.scrollHeight;
  }

  // textarea 자동 크기 조절
  function autoResize() {
    inputText.style.height = 'auto';
    inputText.style.height = inputText.scrollHeight + 'px';
  }

  // 메시지 전송 처리 (fetch + 스트리밍)
  async function sendMessage() {
    const text = inputText.value.trim();
    if (!text) return;

    addMessage(text, true);
    inputText.value = '';
    autoResize();
    sendBtn.disabled = true;

    const botDiv = document.createElement('div');
    botDiv.classList.add('message', 'bot');
    chatContainer.appendChild(botDiv);
    scrollToBottom();

    let fullText = '';
    try {
      // 여기를 너 서버 주소에 맞게 바꿔야 함!
      const encodedMsg = encodeURIComponent(text);
      const url = `https://yuchan5386-elm.hf.space/api/chat?message=${encodedMsg}`;

      const res = await fetch(url, { method: 'GET' });

      if (!res.ok) throw new Error('서버 오류');

      const reader = res.body.getReader();
      const decoder = new TextDecoder('utf-8');

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        chunk.split('\n').forEach(line => {
          if (!line.startsWith('data:')) return;
          const dataStr = line.slice(5).trim();
          if (!dataStr) return;

          try {
            const data = JSON.parse(dataStr);
            if (data.char !== undefined) {
              fullText += data.char;
              botDiv.innerHTML = marked.parse(fullText);
              botDiv.querySelectorAll('pre code').forEach(block => {
                hljs.highlightElement(block);
                makeCopyButton(block);
              });
              scrollToBottom();
            } else if (data.done) {
              botDiv.innerHTML = marked.parse(fullText);
              botDiv.querySelectorAll('pre code').forEach(block => {
                hljs.highlightElement(block);
                makeCopyButton(block);
              });
              scrollToBottom();
            } else if (data.error) {
              botDiv.textContent = '⚠️ ' + data.error;
            }
          } catch (e) {
            console.warn('JSON Parse Error:', e);
          }
        });
      }
    } catch (e) {
      botDiv.textContent = '⚠️ 통신 실패: ' + e.message;
    } finally {
      sendBtn.disabled = false;
      autoResize();
      scrollToBottom();
    }
  }

  // 이벤트 바인딩
  inputText.addEventListener('input', () => {
    autoResize();
    sendBtn.disabled = !inputText.value.trim();
  });

  inputText.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  sendBtn.addEventListener('click', sendMessage);

  $('#new-room-btn').addEventListener('click', () => {
    const name = prompt('새 채팅방 이름은?');
    if (name && createRoom(name)) switchRoom(name);
  });

  // 초기 실행
  init();


