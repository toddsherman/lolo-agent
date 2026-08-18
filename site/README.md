# Write-up site

`index.html` is the self-contained "Adventures of Lolo Agent" write-up: one file, no build
step, no external resource requests. Every figure, the fifty room captures and all run
telemetry are inlined as data URIs.

`lolo-og.png` is the Open Graph share image, referenced absolutely by the page's meta tags.

The page's `canonical`, `og:url` and `twitter:*` tags point at https://www.todd.sh/lolo, which is
the intended home. If it is served anywhere else as its permanent address, update those tags
or link previews will point at the wrong URL.

The room captures in this page are evaluator-only documentation. They were never part of any
training corpus, dataset or agent input path.
