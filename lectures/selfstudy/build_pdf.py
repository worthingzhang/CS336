#!/usr/bin/env python3
"""Interleave unmodified course PDF pages with Chinese self-study explanations.

Install PyMuPDF (tested with 1.28.2), then:
    python build_pdf.py --root /path/to/repo --lecture 3
JSON sources live next to this script. Original slide pages are copied as PDF
objects, never rasterized; validation checks text, geometry, and rendered bytes.
"""
import argparse
import hashlib
import html
import json
import re
import unicodedata
from pathlib import Path

import pymupdf as fitz

CSS = """body { font-family: sans-serif; font-size: 11pt; line-height: 1.65; color: #202b37; }
h1 { font-size: 23pt; line-height: 1.3; color: #173e62; margin-bottom: 18pt; }
h2 { font-size: 15pt; line-height: 1.35; color: #173e62; margin-top: 18pt; }
h3 { font-size: 12pt; color: #173e62; margin-top: 12pt; margin-bottom: 5pt; }
p { margin-top: 4pt; margin-bottom: 9pt; }
.eyebrow { font-size: 10pt; color: #586e80; }
.source { font-size: 9pt; color: #52616d; }
a { color: #205d89; }
"""


def escape(s):
    return html.escape(str(s), quote=True)


def paragraphs(text, cls=''):
    return ''.join(f'<p class="{cls}">{escape(p).replace(chr(10), "<br/>")}</p>'
                   for p in str(text).split('\n\n') if p.strip())


def render_html(markup):
    """Lay out flowing text without truncation; MuPDF embeds fallback CJK fonts."""
    story = fitz.Story(html=markup, user_css=CSS)
    page_rect = fitz.paper_rect('a4')
    content_rect = page_rect + (43, 48, -43, -48)
    def rectangle(number, filled):
        if number > 200:
            raise ValueError('Unexpectedly long section or unplaceable content')
        return page_rect, content_rect, None
    return story.write_with_links(rectangle)


def add_notes(out, markup, label):
    doc = render_html(markup)
    start = len(out)
    for i, page in enumerate(doc):
        page.insert_text((43, 26), label, fontsize=8, color=(.34, .42, .49))
        page.insert_text((43, page.rect.height-25), f'Explanation {i+1}/{len(doc)}', fontsize=8,
                         color=(.34, .42, .49))
    out.insert_pdf(doc)
    doc.close()
    return start, len(out)-start


def compact(text):
    return re.sub(r'[\s\u200b]+', '', unicodedata.normalize('NFKC', text))


def check_note_text(out, start, count, texts, label):
    extracted = compact(''.join(out[p].get_text(clip=out[p].rect+(40,40,-40,-40))
                                for p in range(start, start+count)))
    for text in texts:
        for paragraph in text.split('\n\n'):
            if compact(paragraph) not in extracted:
                raise ValueError(f'Text missing after rendering: {label}: {paragraph[:90]}')


def validate_structure(data, count):
    flat = [p for s in data['sections'] for p in s['pages']]
    if flat != list(range(1, count+1)):
        raise ValueError('sections.pages must cover every original page exactly once in order')
    ids = [r['id'] for r in data['references']]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate reference ids')
    for s in data['sections']:
        if not s['title'] or not s['blocks']:
            raise ValueError('Empty teaching section')
        for block in s['blocks']:
            if not block['text'].strip():
                raise ValueError('Empty teaching block')
        for sid in s.get('sources', []):
            if sid not in ids:
                raise ValueError(f'Unknown source id: {sid}')


def build(root, number):
    lectures = root/'lectures'
    source = lectures/f'lecture_{number:02}.pdf'
    notes = lectures/'selfstudy'/f'lecture_{number:02}.json'
    target = lectures/f'lecture_{number:02}_annotated.pdf'
    data = json.loads(notes.read_text())
    assert data['lecture'] == number
    original_bytes = source.read_bytes()
    original_hash = hashlib.sha256(original_bytes).hexdigest()
    original = fitz.open(source)
    validate_structure(data, len(original))
    refs = {r['id']: r for r in data['references']}
    out = fitz.open()
    toc = []
    intro = (f'<p class="eyebrow">CS336 · Spring 2026 · Lecture {number:02}</p>'
             f'<h1>{escape(data["title"])}</h1>'
             '<h2>如何阅读这份自学增强讲义</h2>'
             '<p>原始英文幻灯片完整保留；每组原页之后紧接对应中文讲解。'
             '书签可跳转到原页与讲解。页码中的“原页”始终指原始 PDF 页码。</p>'
             '<p>中文部分为自学补充；课堂转述、原论文结论与编者推导应按文中标记区分。'
             '第三方保存的字幕仅用于检索定位，不等同于教师审校逐字稿。</p>'
             + ''.join(paragraphs(s) for s in data['intro']))
    intro_start, intro_count = add_notes(out, intro, f'CS336 | Lecture {number:02} | Reading guide')
    check_note_text(out, intro_start, intro_count, [data['title'], *data['intro']], 'intro')
    toc.append([1, f'第 {number} 讲 · 阅读导引', 1])
    mapping = []
    explanation_mapping = []
    for index, section in enumerate(data['sections'], 1):
        first = section['pages'][0]
        last = section['pages'][-1]
        title = section['title']
        toc.append([1, f'{first}–{last} · {title}', len(out)+1])
        for p in section['pages']:
            mapping.append({'original': p, 'output': len(out)+1})
            toc.append([2, f'原页 {p}', len(out)+1])
            out.insert_pdf(original, from_page=p-1, to_page=p-1)
            # insert_pdf does not copy a page's transparency Group. These course
            # PDFs use a direct DeviceRGB group; retaining it is necessary even
            # for exact antialiasing at the fractional page boundary.
            group_type, group_value = original.xref_get_key(original[p-1].xref, 'Group')
            if group_type != 'null':
                if group_type != 'dict' or re.search(r'\d+\s+\d+\s+R\b', group_value):
                    raise ValueError('Indirect transparency groups require explicit resource grafting')
                out.xref_set_key(out[-1].xref, 'Group', group_value)
        markup = (f'<p class="eyebrow">第 {number} 讲 · 原页 {first}–{last} · 中文讲解</p>'
                  f'<h1>{escape(title)}</h1>')
        for b in section['blocks']:
            markup += f'<h3>{escape(b["heading"])}</h3>' + paragraphs(b['text'])
        if section.get('sources'):
            links = []
            for sid in section['sources']:
                r = refs[sid]
                links.append(f'<a href="{escape(r["url"])}">[{escape(sid)}]</a>')
            markup += '<p class="source">本节参考（完整书目见讲义末尾）：' + ' · '.join(links) + '</p>'
        start, count = add_notes(out, markup, f'CS336 | Lecture {number:02} | Original pp. {first}-{last}')
        check_note_text(out, start, count,
                        [section['title']] + [b[k] for b in section['blocks'] for k in ('heading','text')],
                        f'original pp. {first}-{last}')
        explanation_mapping.append({'original_pages': section['pages'],
                                    'output_start': start+1, 'output_count': count})
        toc.append([2, '中文讲解', start+1])
    end = '<h1>自测与参考答案</h1>'
    for i, check in enumerate(data.get('self_checks', []), 1):
        end += f'<h2>{i}. {escape(check["question"])}</h2>' + paragraphs(check['answer'])
    end += '<h1>参考资料</h1>'
    for r in data['references']:
        display_url = escape(r['url']).replace('/', '/&#8203;')
        end += (f'<h3>[{escape(r["id"])}] {escape(r["title"])}</h3>'
                f'<p class="source"><a href="{escape(r["url"])}">{display_url}</a></p>')
    start, count = add_notes(out, end, f'CS336 | Lecture {number:02} | Self-checks and references')
    check_note_text(out, start, count,
                    [q[k] for q in data.get('self_checks',[]) for k in ('question','answer')]
                    + [r[k] for r in data['references'] for k in ('title','url')], 'self-checks and references')
    toc.append([1, '自测与参考资料', start+1])
    out.set_toc(toc)
    out.set_metadata({'title': f'CS336 Lecture {number:02} — {data["title"]} — 中文自学增强版',
                      'author': 'Stanford CS336 original slides; Chinese study annotations',
                      'subject': 'Original slide pages interleaved with Chinese explanations',
                      'keywords': 'CS336,Spring 2026,Chinese,self-study'})
    out.save(target, garbage=4, deflate=True)
    out.close()
    generated = fitz.open(target)
    for entry in mapping:
        before, after = original[entry['original']-1], generated[entry['output']-1]
        assert before.rect == after.rect, entry
        assert before.get_text() == after.get_text(), entry
        assert [original.xref_stream(x) for x in before.get_contents()] == [
            generated.xref_stream(x) for x in after.get_contents()], entry
        mat = fitz.Matrix(.5, .5)
        assert before.get_pixmap(matrix=mat).samples == after.get_pixmap(matrix=mat).samples, entry
    assert source.read_bytes() == original_bytes
    record = {'lecture': number, 'original_sha256': original_hash,
              'notes_sha256': hashlib.sha256(notes.read_bytes()).hexdigest(),
              'original_pages': len(original), 'output_pages': len(generated),
              'explanation_pages': len(generated)-len(original),
              'original_page_mapping': mapping,
              'explanation_page_mapping': explanation_mapping,
              'render_check_scale': 0.5,
              'validation': 'PASS: original bytes, page geometry, content streams, extracted text, rendered pixels, explanation text',
              'renderer': f'PyMuPDF {fitz.VersionBind}'}
    report = lectures/'selfstudy'/f'lecture_{number:02}.build.json'
    report.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k: v for k, v in record.items() if not k.endswith('page_mapping')},ensure_ascii=False,indent=2))
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--lecture', type=int, required=True)
    args = parser.parse_args()
    build(args.root, args.lecture)
