import { enableVisualEditing } from "@sanity/visual-editing-standalone";

function historyAdapter() {
  return {
    subscribe(navigate) {
      const onPopState = () => {
        navigate({ type: "pop", url: window.location.href });
      };

      window.addEventListener("popstate", onPopState);
      return () => window.removeEventListener("popstate", onPopState);
    },
    update(update) {
      if (update.type === "push") {
        window.history.pushState(null, "", update.url);
      } else if (update.type === "replace") {
        window.history.replaceState(null, "", update.url);
      } else if (update.type === "pop") {
        window.history.back();
      }
    },
  };
}

async function refreshPreview(payload) {
  if (
    payload.source === "mutation" &&
    payload.document?._type !== "loloPage"
  ) {
    return false;
  }

  if (typeof window.toddCmsRefresh === "function") {
    await window.toddCmsRefresh();
    return;
  }

  window.location.reload();
}

if (window.parent !== window) {
  Promise.resolve(window.toddCmsReady).finally(() => {
    enableVisualEditing({
      history: historyAdapter(),
      refresh: refreshPreview,
      zIndex: 1000,
    });
  });
}
