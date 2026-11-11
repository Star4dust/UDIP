from pathlib import Path
import re

root = Path('docs')
output = root / 'diagrams'
output.mkdir(parents=True, exist_ok=True)

for md in root.glob('*.md'):
    text = md.read_text(encoding='utf-8')
    blocks = re.split(r'```mermaid\n', text)
    for idx, block in enumerate(blocks[1:], start=1):
        if '```' in block:
            diagram, _ = block.split('```', 1)
        else:
            diagram = block
        name = md.stem if idx == 1 else f'{md.stem}_{idx}'
        out_path = output / f'{name}.mmd'
        out_path.write_text(diagram.strip() + '\n', encoding='utf-8')
        print(f'Wrote {out_path}')
