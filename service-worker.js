// 루트 또는 public/ 아래에 service-worker.js
const CACHE_NAME = 'covec-cache-v8';
const urlsToCache = [
  '/',
  '/CoVec-Web/index.html',
  '/CoVec-Web/css/style.css',
  '/CoVec-Web/js/script.js',
  '/CoVec-Web/manifest.json',
  '/CoVec-Web/icon-192.png',
  '/CoVec-Web/icon-512.png',
  'https://cdn.jsdelivr.net/npm/marked/marked.min.js',
  'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.8.0/build/highlight.min.js',
  'https://cdn.jsdelivr.net/gh/highlightjs/cdn-release@11.8.0/build/styles/atom-one-dark.min.css'
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
