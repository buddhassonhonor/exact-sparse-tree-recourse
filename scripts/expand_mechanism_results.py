from pathlib import Path
import gzip,shutil
root=Path(__file__).resolve().parents[1]
for source in (root/'exp').rglob('*.csv.gz'):
    with gzip.open(source,'rb') as inp, source.with_suffix('').open('wb') as out:
        shutil.copyfileobj(inp,out)
print('Expanded archived result tables.')
