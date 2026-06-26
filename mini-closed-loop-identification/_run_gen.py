
import os

def w(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    lines = content.count(chr(10))
    print(f'  {path}: {lines} lines')
    return lines

total = 0

# ─── clid_direct.c (core ARX + ARMAX) ───
total += w('src/clid_direct.c', open('_direct_c.txt','r',encoding='utf-8').read())
print(f'Total lines so far: {total}')
