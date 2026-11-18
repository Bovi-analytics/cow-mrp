/* CowMRP website: theme switch, copy buttons, outline highlight, page search. Everything is local; nothing is sent anywhere. */
(function () {
  var root = document.documentElement;
  try { var t = localStorage.getItem("cowmrp-theme"); if (t) root.setAttribute("data-theme", t); } catch (e) {}
  var btn = document.getElementById("theme");
  if (btn) btn.onclick = function () {
    var dark = root.getAttribute("data-theme") === "dark" ||
      (!root.getAttribute("data-theme") && window.matchMedia("(prefers-color-scheme: dark)").matches);
    var next = dark ? "light" : "dark";
    root.setAttribute("data-theme", next);
    try { localStorage.setItem("cowmrp-theme", next); } catch (e) {}
  };

  document.querySelectorAll("pre").forEach(function (p) {
    if (p.querySelector("button.copy") || !p.querySelector("code")) return;
    var b = document.createElement("button"); b.className = "copy"; b.textContent = "Copy";
    b.onclick = function () {
      var txt = p.querySelector("code").innerText;
      (navigator.clipboard ? navigator.clipboard.writeText(txt) : Promise.reject()).then(function () {
        b.textContent = "Copied"; setTimeout(function () { b.textContent = "Copy"; }, 1200);
      }).catch(function () {});
    };
    p.appendChild(b);
  });

  var links = document.querySelectorAll(".outline a");
  if (links.length && "IntersectionObserver" in window) {
    var map = {}; links.forEach(function (a) { map[a.getAttribute("href").slice(1)] = a; });
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) { if (e.isIntersecting) { links.forEach(function (a) { a.classList.remove("cur"); }); var a = map[e.target.id]; if (a) a.classList.add("cur"); } });
    }, { rootMargin: "-70px 0px -70% 0px" });
    Object.keys(map).forEach(function (id) { var el = document.getElementById(id); if (el) io.observe(el); });
  }

  var q = document.getElementById("q"), box = document.getElementById("results");
  if (q && box) {
    var index = null, base = q.getAttribute("data-base") || "";
    function load(cb) {
      if (index) return cb();
      var s = document.createElement("script"); s.src = base + "assets/search-index.js";
      s.onload = function () { index = window.COWMRP_SEARCH || []; cb(); }; document.head.appendChild(s);
    }
    q.addEventListener("input", function () {
      var v = q.value.trim().toLowerCase();
      if (v.length < 2) { box.style.display = "none"; return; }
      load(function () {
        var words = v.split(/\s+/), hits = [];
        index.forEach(function (d) {
          var hay = (d.t + " " + d.h + " " + d.x).toLowerCase(), score = 0;
          for (var i = 0; i < words.length; i++) { if (hay.indexOf(words[i]) < 0) return; if (d.t.toLowerCase().indexOf(words[i]) >= 0) score += 3; if (d.h.toLowerCase().indexOf(words[i]) >= 0) score += 2; score += 1; }
          hits.push([score, d]);
        });
        hits.sort(function (a, b) { return b[0] - a[0]; });
        box.innerHTML = hits.slice(0, 8).map(function (h) { return '<a href="' + base + h[1].u + '"><b>' + h[1].t + "</b><small>" + h[1].h.split(" | ").slice(0, 3).join(" · ") + "</small></a>"; }).join("") || '<a><small>No match</small></a>';
        box.style.display = "block";
      });
    });
    document.addEventListener("click", function (e) { if (!box.contains(e.target) && e.target !== q) box.style.display = "none"; });
  }
})();
