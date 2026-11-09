# Write-up site

`index.html` is the static "Adventures of Lolo Agent" write-up: no build step and no
third-party resource requests. Every telemetry figure and the fifty room captures are
inlined as data URIs; the supporting hero image is the local `lolo-hero.webp` asset.

`lolo-og.png` is the Open Graph share image, referenced absolutely by the page's meta tags.
It and `lolo-hero.webp` use the same Todd dot sh palette artwork.

The page's `canonical`, `og:url` and `twitter:*` tags point at https://www.todd.sh/lolo, which is
the intended home. If it is served anywhere else as its permanent address, update those tags
or link previews will point at the wrong URL.

The room captures in this page are evaluator-only documentation. They were never part of any
training corpus, dataset or agent input path.
