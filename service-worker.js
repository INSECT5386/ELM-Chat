// service-worker.js
self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open("covec-cache-v1").then((cache) => {
      return cache.addAll([
        "/",
        "/index.html",
        "/css/style.css",
        "/js/script.js",
        "/manifest.json",
        "/icon-192.png",
        "/icon-512.png",
      ]);
    })
  );
});

self.addEventListener("fetch", (e) => {
  e.respondWith(
    caches.match(e.request).then((res) => {
      return res || fetch(e.request);
    })
  );
});
