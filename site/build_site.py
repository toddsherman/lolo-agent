#!/usr/bin/env python3
"""Build the self-contained write-up page from lolo-shell.html + writeup/payload.json.

Output is a complete HTML document (the artifact host injects doctype/head/reset;
a plain static host does not). Run this, then copy to the lolo-agent repo's site/.
"""
import json, os, re

S = os.path.dirname(os.path.abspath(__file__))
SITE = 'https://www.todd.sh/lolo'          # canonical: todd.sh 308s to www
IMG  = SITE + '/lolo-og.png'               # resolves through the /lolo/:path* rewrite

shell = open(f'{S}/lolo-shell.html').read()
payload = open(f'{S}/writeup/payload.json').read().replace('—', ', ').replace('</', '<\\/')
body = shell.replace('__PAYLOAD__', payload)

m = re.match(r'\s*<title>(.*?)</title>\s*', body, re.S)
title, body = m.group(1), body[m.end():]

DESC = ("An agent that learns NES Adventures of Lolo from pixels, a controller and save states "
        "alone, with no rules, object names, solutions or demonstrations. Field notes on what "
        "failed, four times the project proved itself wrong, and the one heart it finally collected.")
FAV = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E"
       "%3Ctext y='.9em' font-size='90'%3E%F0%9F%95%B9%EF%B8%8F%3C/text%3E%3C/svg%3E")

head = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{DESC}">
<meta name="author" content="Todd Sherman">
<link rel="canonical" href="{SITE}">
<link rel="icon" href="{FAV}">
<meta property="og:type" content="article">
<meta property="og:site_name" content="todd.sh">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{DESC}">
<meta property="og:url" content="{SITE}">
<meta property="og:image" content="{IMG}">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="Rooms of Adventures of Lolo captured from the emulator used in this project">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{title}">
<meta name="twitter:description" content="{DESC}">
<meta name="twitter:image" content="{IMG}">
<meta name="theme-color" content="#f6f1e6" media="(prefers-color-scheme: light)">
<meta name="theme-color" content="#141110" media="(prefers-color-scheme: dark)">
<style>
/* minimal reset: the artifact host supplied one, a plain web server does not */
*,*::before,*::after{{box-sizing:border-box}}
html{{-webkit-text-size-adjust:100%}}
img,svg,canvas{{max-width:100%}}
</style>
'''
out = head + body.rstrip() + '\n</body>\n</html>\n'
out = out.replace('</style>\n\n<div class="bleed"',
                  '</style>\n</head>\n<body>\n\n<div class="bleed"', 1)
open(f'{S}/lolo-standalone.html', 'w').write(out)

d = json.loads(re.search(r'<script id="payload" type="application/json">(.*?)</script>', out, re.S).group(1))
open(f'{S}/app.js', 'w').write(re.search(r'<script>\n(const D=JSON.parse.*?)</script>', out, re.S).group(1))
toks = set(re.findall(r'var\(--([a-z0-9-]+)\)', out))
undef = sorted(toks - set(re.findall(r'--([a-z0-9-]+):', out.split('@media', 1)[0])))
assert out.startswith('<!doctype html>') and out.count('<body>') == 1 and out.count('</head>') == 1
assert out.count('—') == 0 and not undef, (out.count('—'), undef)
assert not re.search(r'src="https?://', out), 'external resource load'
print(f'built {os.path.getsize(S+"/lolo-standalone.html")/1024:.1f}KB  runs={len(d["runs"])} '
      f'rooms={d["allrooms"]["captured"]}  canonical={SITE}  em-dashes=0  undefined-tokens=none')
