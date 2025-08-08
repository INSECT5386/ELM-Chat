// 루트 또는 public/ 아래에 service-worker.js
const CACHE_NAME = 'covec-cache-v15';
const urlsToCache = [
  '/ELM-Chat/',
  '/ELM-Chat/index.html',
  '/ELM-Chat/css/style.css',
  '/ELM-Chat/js/script.js',
  '/ELM-Chat/manifest.json',
  '/ELM-Chat/img/icon-192.png',
  '/ELM-Chat/img/icon-512.png',
  'https://cdn.jsdelivr.net/npm/marked/marked.min.js',
  'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.8.0/build/highlight.min.js',
  'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.8.0/build/styles/atom-one-light.min.css'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      console.log('📦 캐시 저장 시작');
      return cache.addAll(urlsToCache);
    })
  );
});

self.addEventListener('fetch', (event) => {
  event.respondWith(
    caches.match(event.request).then(response => {
      // 캐시에 있으면 반환, 없으면 네트워크 요청
      return response || fetch(event.request);
    })
  );
});
