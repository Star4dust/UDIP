import os
paths = [
    'node_modules/.bin',
    r'node_modules\@mermaid-js',
    r'node_modules\@mermaid-js\mermaid-cli',
    'node_modules/@mermaid-js',
    'node_modules/@mermaid-js/mermaid-cli'
]
for path in paths:
    print('PATH:', path)
    print('exists:', os.path.exists(path))
    print('isdir:', os.path.isdir(path))
    if os.path.isdir(path):
        print('entries:', os.listdir(path)[:20])
    print('----')
