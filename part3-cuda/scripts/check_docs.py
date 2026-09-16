from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlparse,unquote
import json,re
R=Path(__file__).resolve().parents[1]
class Links(HTMLParser):
 def __init__(self):super().__init__();self.refs=[]
 def handle_starttag(self,tag,attrs):
  for k,v in attrs:
   if k in ('href','src') and v:self.refs.append(v)
issues=[];pages=0;line_count=0
for p in (R/'docs').glob('*.html'):
 pages+=1;s=p.read_text();parser=Links();parser.feed(s)
 for link in parser.refs:
  u=urlparse(link)
  if u.scheme or link.startswith(('/', '#')):continue
  target=(p.parent/unquote(u.path)).resolve()
  if not target.exists():issues.append(f'{p.name}: missing {link}')
 if p.name.startswith('v'):
  data=json.loads(re.search(r'<script id="data" type="application/json">(.*?)</script>',s,re.S)[1]);source=(R/'kernels'/p.with_suffix('.cu').name).read_text().splitlines()
  expected={i:l for i,l in enumerate(source,1) if l.strip()}
  found={x['line']:x['code'] for x in data['lines']}
  if found!=expected:issues.append(p.name+': line mapping mismatch')
  if not all(x['note'] for x in data['lines']):issues.append(p.name+': empty explanation')
  line_count+=len(found)
 for ptn in ['__TITLE__','__DATA__','__INTRO__','__SVG__','__LAB_JS__']:
  if ptn in s:issues.append(p.name+': unresolved '+ptn)
report={'html_pages':pages,'documented_source_lines':line_count,'issues':issues}
(R/'results/doc-validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False))
if issues:raise SystemExit(1)
