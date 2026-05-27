import glob, re, sys
labels = set()
refs = set()
for f in glob.glob('latex_project/thesis/*.tex'):
    text = open(f, encoding='utf-8').read()
    text = re.sub(r'(?m)%.*$', '', text)
    for m in re.finditer(r'\\label\{([^}]+)\}', text):
        labels.add(m.group(1))
    for m in re.finditer(r'\\ref\{([^}]+)\}', text):
        refs.add(m.group(1))
print('refs without matching label:', sorted(refs - labels))
print('labels never referenced  :', sorted(labels - refs))
