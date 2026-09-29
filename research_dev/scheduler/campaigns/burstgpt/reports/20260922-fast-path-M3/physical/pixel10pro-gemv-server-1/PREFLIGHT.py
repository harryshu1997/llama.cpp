import hashlib,json,subprocess
from pathlib import Path
config=json.loads(Path("PIXEL_GEMV_SERVER_CONFIG.json").read_text())
processes=subprocess.check_output(["ps","-eo","pid,comm,args"],text=True)
active=[line for line in processes.splitlines()[1:] if len(line.split())>1 and line.split()[1].startswith("llama-")]
gpu=subprocess.check_output(["nvidia-smi","--query-compute-apps=pid,process_name","--format=csv,noheader"],text=True)
assert not active, active
assert not gpu.strip(),gpu
expected={
"llama-ffn-split-worker":"7cf01c7ae4940a92bd44d7d2b1b4a5bac0fe0c02d2953c8ef39158aa398d03ad",
"libggml.so":"601b8a7c7d14ab951ee85e6680bea3ed428b73ed63fc8a1fcb55d4999760c6f2",
"libggml-base.so":"ec8396655c0b24bf828702a6e83371fe6bd3f24b57149e3be7d666813ff9f49d",
"libggml-cpu.so":"6ca5bcc4948428833216b2b16c885e87aaffc15e45cbef8e6fe3d8069fc39250",
"libggml-vulkan.so":"a97cb05dcac71b8adf826559e27ee62b9286c3c4a571523d5583976b60856b22"}
files=[config["phone_worker"],*config["phone_libraries"]]
text=subprocess.check_output(["adb","-P","5037","-s",config["serial"],"shell","sha256sum",*files],text=True,timeout=60)
for line in text.splitlines():
    digest,path=line.split()
    assert expected[Path(path).name]==digest,(path,digest)
assert len(text.splitlines())==len(expected)
lib=Path(config["server"]).parent/"libllama-server-impl.so"
strings=subprocess.check_output(["strings",str(lib)],text=True)
count=sum("S41SERVERFFN" in line for line in strings.splitlines())
assert count>=12,count
with Path("PREFLIGHT.json").open("x") as f:
    json.dump({"status":"PASS","processes":processes,"gpu_processes":gpu,"phone_hashes":text,"ffn_marker_lines":count,"server_impl_sha256":hashlib.sha256(lib.read_bytes()).hexdigest()},f,indent=2)
