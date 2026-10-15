#!/usr/bin/env python3
"""Verify that Chinese lecture annotations preserve original Python behavior."""
import argparse
import ast
import io
import json
from pathlib import Path
import subprocess
import tokenize


def tokens(source):
    ignored = {tokenize.COMMENT, tokenize.NL, tokenize.ENCODING}
    return [(t.type, t.string) for t in tokenize.generate_tokens(io.StringIO(source).readline)
            if t.type not in ignored]


def verify(root, number, baseline):
    path = f'lectures/lecture_{number:02}.py'
    original = subprocess.check_output(['git', 'show', f'{baseline}:{path}'], cwd=root).decode()
    current = (root/path).read_text()
    assert ast.dump(ast.parse(original)) == ast.dump(ast.parse(current)), 'AST changed'
    assert tokens(original) == tokens(current), 'Executable tokens changed'
    it = iter(current.splitlines(keepends=True))
    for line in original.splitlines(keepends=True):
        assert any(line == candidate for candidate in it), 'An original line was removed or altered'
    compile(current, path, 'exec')
    diff = subprocess.run(['git', 'diff', '--check', '--', path], cwd=root,
                          capture_output=True, text=True)
    assert diff.returncode == 0, diff.stdout + diff.stderr
    added_lines = len(current.splitlines())-len(original.splitlines())
    added_cjk = sum('\u4e00' <= c <= '\u9fff' for c in current)-sum('\u4e00' <= c <= '\u9fff' for c in original)
    assert added_cjk > 0, 'No new Chinese annotations'
    result = {'lecture': number, 'file': path, 'baseline': baseline,
              'original_lines': len(original.splitlines()), 'annotated_lines': len(current.splitlines()),
              'added_lines': added_lines, 'added_cjk_characters': added_cjk,
              'syntax': 'PASS', 'ast': 'IDENTICAL', 'executable_tokens': 'IDENTICAL',
              'original_lines_in_order': 'PASS', 'diff_check': 'PASS',
              'runtime': 'Not covered by this static check'}
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument('--lecture', type=int, required=True)
    p.add_argument('--baseline', default='d59bbebb257c01b090e40bf60e2c0f960df67d05')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    result = json.dumps(verify(a.root, a.lecture, a.baseline), ensure_ascii=False, indent=2)+'\n'
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(result)
    print(result)
