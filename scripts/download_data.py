"""Official direct downloads. No auth, scraping or silent synthetic fallback."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
URLS={'cmapss':'https://data.nasa.gov/docs/legacy/CMAPSSData.zip',
      'ai4i':'https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip'}


def download(name):
    url=URLS[name]
    request=Request(url,headers={'User-Agent':'TimeGuardAI-research/1.0'})
    with urlopen(request,timeout=120) as response:
        data=response.read()
    folder=ROOT/'data'/'raw'/name
    folder.mkdir(parents=True,exist_ok=True)
    expected=['train_FD001.txt','test_FD001.txt','RUL_FD001.txt'] if name=='cmapss' else ['ai4i2020.csv']
    # Extract only known basenames, preventing arbitrary archive paths.
    with ZipFile(io.BytesIO(data)) as archive:
        for basename in expected:
            matches=[n for n in archive.namelist() if Path(n).name==basename]
            if len(matches)!=1:
                raise ValueError(f'Archive must contain exactly one {basename}')
            (folder/basename).write_bytes(archive.read(matches[0]))
    (folder/'download.zip').write_bytes(data)
    (folder/'provenance.json').write_text(json.dumps({'source':url,'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)},indent=2))
    print(f'{name}: validated {len(expected)} files, {len(data):,} bytes')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['all',*URLS],default='all')
    args=parser.parse_args()
    for name in URLS if args.dataset=='all' else [args.dataset]:
        download(name)
