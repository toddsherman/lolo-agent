# Write-up site

`index.html` is the static "Adventures of Lolo Agent" write-up: no build step and no
third-party resource requests. Every telemetry figure and the fifty room captures are
inlined as data URIs; the supporting hero image is the local `lolo-hero.webp` asset.

`lolo-og.png` is the Open Graph share image, referenced absolutely by the page's meta tags.
It and `lolo-hero.webp` use the same Todd dot sh palette artwork.

`cms-copy.js` overlays published wording from the shared todd.sh Sanity project onto
fixed text positions. If Sanity is unavailable, the checked-in wording in `index.html`
remains visible. The script changes text and accessibility descriptions only; it does
not accept HTML or alter the page structure, links, figures, interactions, or styles.

When the page is opened inside Sanity's Presentation tool, it fetches draft wording
through todd.sh's protected preview endpoint and adds click-to-edit targets. The visual
editing bundle is built from `../site-tools/` and is loaded only inside an iframe, so
regular visitors do not download the editor runtime.

The page's `canonical`, `og:url` and `twitter:*` tags point at https://www.todd.sh/lolo, which is
the intended home. If it is served anywhere else as its permanent address, update those tags
or link previews will point at the wrong URL.

The room captures in this page are evaluator-only documentation. They were never part of any
training corpus, dataset or agent input path.
