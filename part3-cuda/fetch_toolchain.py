#!/usr/bin/env python3
"""Download SHA256-pinned NVIDIA CUDA redistributables into this workspace only."""
import argparse,concurrent.futures,hashlib,json,pathlib,subprocess,tarfile
project=pathlib.Path(__file__).resolve().parent
root=project.parents[1]
manifest=json.loads((project/'results/cuda-13.0.2-components.json').read_text())
p=argparse.ArgumentParser();p.add_argument('components',nargs='*',default=['cuda_nvcc','cuda_cudart','cuda_cccl','cuda_crt','libnvvm','libcublas']);args=p.parse_args()
def fetch(k):
 e=manifest[k]['linux-x86_64'];dest=root/'downloads'/pathlib.Path(e['relative_path']).name;dest.parent.mkdir(exist_ok=True)
 def valid():return dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest()==e['sha256']
 if not valid():
  print('DOWNLOAD',k,flush=True)
  subprocess.run(['curl','-L','--fail','--retry','2','--max-time','600','-sS','https://developer.download.nvidia.com/compute/cuda/redist/'+e['relative_path'],'-o',str(dest)],check=True)
 if not valid():raise RuntimeError('SHA256 mismatch: '+k)
 with tarfile.open(dest) as t:
  for m in t.getmembers():
   m.name='/'.join(m.name.split('/')[1:])
   if m.name:t.extract(m,root/'.toolchains/cuda-13.0')
 print('READY',k,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:list(ex.map(fetch,args.components))
