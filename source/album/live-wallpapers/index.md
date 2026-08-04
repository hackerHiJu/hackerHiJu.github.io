---
title: 收藏的壁纸
date: 2025-08-03 18:00:00
top_img: false
aside: false
---

{% raw %}
<style>
.wp-gallery {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
  gap: 20px;
  padding: 2rem 0;
  max-width: 1200px;
  margin: 0 auto;
}
.wp-card {
  position: relative;
  border-radius: 16px;
  overflow: hidden;
  cursor: pointer;
  aspect-ratio: 9 / 16;
  box-shadow: 0 4px 20px rgba(0,0,0,0.1);
  transition: transform 0.3s, box-shadow 0.3s;
}
.wp-card:hover {
  transform: translateY(-6px);
  box-shadow: 0 12px 36px rgba(0,0,0,0.2);
}
.wp-card img,
.wp-card video {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
  pointer-events: none;
}
.wp-card .wp-overlay {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 12px 16px;
  background: linear-gradient(transparent, rgba(0,0,0,0.7));
  color: #fff;
  font-size: 0.9rem;
  font-weight: 500;
  opacity: 0;
  transition: opacity 0.3s;
}
.wp-card:hover .wp-overlay {
  opacity: 1;
}
.wp-card .wp-zoom {
  position: absolute;
  top: 12px;
  right: 12px;
  width: 36px;
  height: 36px;
  background: rgba(0,0,0,0.5);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  font-size: 16px;
  opacity: 0;
  transition: opacity 0.3s;
  z-index: 2;
}
.wp-card:hover .wp-zoom {
  opacity: 1;
}
@media (max-width: 768px) {
  .wp-gallery {
    grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
    gap: 12px;
    padding: 1rem;
  }
}
</style>

<div class="wp-gallery">

  <div class="wp-card no-lightbox" data-src="/album/live-wallpapers/bookshelf.mp4" data-type="video" data-caption="📚 书架书桌凉风">
    <span class="wp-zoom">🔍</span>
    <video class="nolazyload no-lightbox" loop muted autoplay playsinline>
      <source src="/album/live-wallpapers/bookshelf.mp4" type="video/mp4">
    </video>
    <div class="wp-overlay">📚 书架书桌凉风</div>
  </div>

  <div class="wp-card no-lightbox" data-src="/album/live-wallpapers/cloud-mountain-sunset.png" data-type="image" data-caption="🏔️ 云朵山脉日落">
    <span class="wp-zoom">🔍</span>
    <img class="nolazyload no-lightbox" src="/album/live-wallpapers/cloud-mountain-sunset.png" alt="云朵山脉日落">
    <div class="wp-overlay">🏔️ 云朵山脉日落</div>
  </div>

</div>

<script>
function initWpGallery() {
  if (typeof Fancybox === 'undefined') {
    setTimeout(initWpGallery, 200);
    return;
  }
  var cards = document.querySelectorAll('.wp-card');
  cards.forEach(function(card) {
    if (card.dataset.bound) return;
    card.dataset.bound = '1';
    card.addEventListener('click', function(e) {
      e.preventDefault();
      e.stopPropagation();
      var src = card.getAttribute('data-src');
      var caption = card.getAttribute('data-caption');
      Fancybox.show(
        [{ src: src, caption: caption }],
        {
          Hash: false,
          Thumbs: { autoStart: false },
          Toolbar: {
            display: {
              left: ["infobar"],
              middle: [],
              right: ["slideshow", "fullscreen", "thumbs", "download", "close"],
            },
          },
        }
      );
    });
  });
}
document.addEventListener("DOMContentLoaded", initWpGallery);
document.addEventListener("pjax:complete", initWpGallery);
</script>
{% endraw %}
