import os, textwrap
def w(p,c):
    os.makedirs(os.path.dirname(p),exist_ok=True)
    c=textwrap.dedent(c)
    with open(p,"w",encoding="utf-8") as f: f.write(c)
    return c.count(chr(10))
T=0
