(function loadToddShCopy() {
  "use strict";

  const projectId = "8ddex1ni";
  const dataset = "production";
  const apiVersion = "2026-08-21";
  const documentId = "page-lolo";
  const documentType = "loloPage";
  const isVisualPreview = window.parent !== window;

  function groupFor(element) {
    if (element.closest("header.project-hero")) return "intro";
    if (element.closest("section.results")) return "ledger";
    if (element.closest("nav.contents")) return "contents";

    const section = element.closest("section[id]");
    if (section) {
      return section.id.replace(/[^a-z0-9]+/gi, "_").toLowerCase();
    }

    if (element.closest("footer")) return "footer";
    return "site";
  }

  function editableTextNodes() {
    const walker = document.createTreeWalker(
      document.body,
      NodeFilter.SHOW_TEXT,
      {
        acceptNode(node) {
          if (!node.nodeValue || !node.nodeValue.replace(/\s+/g, " ").trim()) {
            return NodeFilter.FILTER_REJECT;
          }

          const parent = node.parentElement;
          if (
            !parent ||
            parent.closest(
              'script,style,noscript,template,svg,canvas,[aria-hidden="true"]',
            )
          ) {
            return NodeFilter.FILTER_REJECT;
          }

          return NodeFilter.FILTER_ACCEPT;
        },
      },
    );

    const counts = Object.create(null);
    const nodes = [];
    let node;

    while ((node = walker.nextNode())) {
      const group = groupFor(node.parentElement);
      counts[group] = (counts[group] || 0) + 1;
      nodes.push({
        key: `${group}_${String(counts[group]).padStart(3, "0")}`,
        node,
      });
    }

    return nodes;
  }

  function setTextNode(node, value) {
    if (typeof value !== "string" || !value.trim()) return;

    const original = node.nodeValue || "";
    const leading = original.match(/^\s*/)?.[0] || "";
    const trailing = original.match(/\s*$/)?.[0] || "";
    node.nodeValue = `${leading}${value.trim()}${trailing}`;
  }

  function setMeta(selector, value) {
    if (typeof value !== "string" || !value.trim()) return;
    document.querySelector(selector)?.setAttribute("content", value.trim());
  }

  function dataSanity(path) {
    const studioUrl = `${window.location.origin}/admin`;
    return [
      `id=${documentId}`,
      `type=${documentType}`,
      `path=${path}`,
      `base=${encodeURIComponent(studioUrl)}`,
      "tool=presentation",
    ].join(";");
  }

  function addVisualEditingTargets(textNodes) {
    if (!isVisualPreview) return;

    for (const { key, node } of textNodes) {
      if (!node.parentNode || node.parentElement?.closest("#allrooms-cap")) {
        continue;
      }

      const target = document.createElement("span");
      target.setAttribute("data-sanity", dataSanity(key));
      node.parentNode.insertBefore(target, node);
      target.appendChild(node);
    }

    document
      .querySelector(".hero-art img")
      ?.setAttribute("data-sanity", dataSanity("heroImageAlt"));
    document
      .getElementById("allrooms-cap")
      ?.setAttribute("data-sanity", dataSanity("roomGalleryCaption"));
  }

  async function fetchPublicContent() {
    const query = encodeURIComponent(`*[_id == "${documentId}"][0]`);
    const url =
      `https://${projectId}.api.sanity.io/v${apiVersion}/data/query/` +
      `${dataset}?perspective=published&query=${query}`;
    const response = await fetch(url, { cache: "no-store" });

    if (!response.ok) throw new Error(`Sanity returned ${response.status}`);
    return response.json();
  }

  async function fetchContent() {
    if (
      window.location.pathname === "/lolo" ||
      window.location.pathname.startsWith("/lolo/")
    ) {
      try {
        const response = await fetch("/api/lolo-content", {
          cache: "no-store",
          credentials: "same-origin",
        });

        if (response.ok) return response.json();
      } catch {
        // Fall through to the public published-content endpoint.
      }
    }

    return fetchPublicContent();
  }

  function applyContent(result, textNodes) {
    if (!result) return false;

    for (const { key, node } of textNodes) {
      setTextNode(node, result[key]);
    }

    if (typeof result.browserTitle === "string" && result.browserTitle.trim()) {
      document.title = result.browserTitle.trim();
      setMeta('meta[property="og:title"]', result.browserTitle);
      setMeta('meta[name="twitter:title"]', result.browserTitle);
    }

    setMeta('meta[name="description"]', result.metaDescription);
    setMeta('meta[property="og:description"]', result.metaDescription);
    setMeta('meta[name="twitter:description"]', result.metaDescription);

    if (typeof result.heroImageAlt === "string" && result.heroImageAlt.trim()) {
      document
        .querySelector(".hero-art img")
        ?.setAttribute("alt", result.heroImageAlt.trim());
    }

    if (
      typeof result.roomGalleryCaption === "string" &&
      result.roomGalleryCaption.trim()
    ) {
      const caption = document.getElementById("allrooms-cap");
      if (caption) caption.textContent = result.roomGalleryCaption.trim();
    }

    document.documentElement.dataset.cmsContent = "loaded";
    return true;
  }

  const textNodes = editableTextNodes();
  addVisualEditingTargets(textNodes);

  window.toddCmsRefresh = () =>
    fetchContent().then(({ result }) => applyContent(result, textNodes));
  window.toddCmsReady = window.toddCmsRefresh().catch(() => false);
})();
