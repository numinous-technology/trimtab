#!/bin/bash
# Run the M1 bench on a RunPod secure-cloud pod, then terminate it.
#
#   bench/launch_runpod.sh h100 sglang
#   bench/launch_runpod.sh h100 vllm
#   bench/launch_runpod.sh rtx6000 sglang
#
# Budget guard. The pod is created with a $/hr ceiling and is removed on exit
# no matter how the run ends. MAX_MIN caps wall time.
set -u
TARGET=${1:?h100|rtx6000}; ENGINE=${2:?sglang|vllm}
MODEL=${MODEL:-Qwen/Qwen3.8-27B-FP8}; MAX_MIN=${MAX_MIN:-75}
case $TARGET in
  h100)    GPU="NVIDIA H100 80GB HBM3"; COST=3.5 ;;
  rtx6000) GPU="NVIDIA RTX PRO 6000 Blackwell Server Edition"; COST=2.5 ;;
  *) echo "unknown target"; exit 2 ;;
esac
IMAGE=$([ $ENGINE = vllm ] && echo "vllm/vllm-openai:latest" || echo "lmsysorg/sglang:latest")
HERE=$(cd "$(dirname "$0")/.." && pwd)
NAME=trimtab-$TARGET-$ENGINE
PUB=$(cat ~/.ssh/id_ed25519.pub)
RES=$HERE/bench/results; mkdir -p $RES

OUT=$(runpodctl create pod --name $NAME --secureCloud --gpuType "$GPU" --imageName "$IMAGE" \
  --containerDiskSize 60 --volumeSize 100 --volumePath /workspace --ports "22/tcp" --cost $COST --mem 64 --vcpu 12 \
  --args "bash -c 'apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq openssh-server >/dev/null 2>&1; mkdir -p /run/sshd /root/.ssh; echo $PUB > /root/.ssh/authorized_keys; chmod 600 /root/.ssh/authorized_keys; /usr/sbin/sshd -D'" 2>&1 | tail -1)
echo "$OUT"
POD=$(echo "$OUT" | grep -oP 'pod "\K[^"]+') || { echo "create failed"; exit 1; }
trap 'echo "removing pod $POD"; runpodctl remove pod $POD >/dev/null 2>&1' EXIT
T0=$(date +%s)

ssh_target(){ runpodctl pod list 2>/dev/null | python3 -c "
import json,sys
for p in json.load(sys.stdin):
    if p['id']=='$POD':
        for x in (p.get('runtime') or {}).get('ports') or []:
            if x.get('privatePort')==22: print(x['ip'], x['publicPort'])"; }
until read -r IP PORT < <(ssh_target) && [ -n "${IP:-}" ] && ssh -o StrictHostKeyChecking=no -o ConnectTimeout=8 -p $PORT root@$IP true 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt 900 ] && { echo "ssh never came up"; exit 1; }; sleep 15
done
S="ssh -o StrictHostKeyChecking=no -p $PORT root@$IP"
echo "ssh up at $IP:$PORT after $(( $(date +%s) - T0 ))s"
$S "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"

tar -C $HERE -czf /tmp/trimtab.tgz --exclude .git --exclude bench/results . && scp -o StrictHostKeyChecking=no -P $PORT /tmp/trimtab.tgz root@$IP:/root/ >/dev/null
$S "mkdir -p /root/trimtab && tar -C /root/trimtab -xzf /root/trimtab.tgz && cd /root/trimtab && \
    ENGINE=$ENGINE MODEL=$MODEL PYTHONPATH=/root/trimtab nohup bash bench/pod_run.sh > /root/pod_run.log 2>&1 &"

while ! $S "test -f /root/trimtab_results/$ENGINE.json" 2>/dev/null; do
  [ $(( $(date +%s) - T0 )) -gt $(( MAX_MIN * 60 )) ] && { echo "TIMEOUT after $MAX_MIN min"; $S "tail -40 /root/pod_run.log"; exit 1; }
  $S "grep -qE 'FAILED|TIMEOUT' /root/pod_run.log" 2>/dev/null && { echo "pod run failed"; $S "tail -60 /root/pod_run.log"; exit 1; }
  sleep 30
done
scp -o StrictHostKeyChecking=no -P $PORT root@$IP:/root/trimtab_results/$ENGINE.json $RES/$TARGET-$ENGINE.json >/dev/null
scp -o StrictHostKeyChecking=no -P $PORT root@$IP:/root/pod_run.log $RES/$TARGET-$ENGINE.log >/dev/null
echo "results in $RES/$TARGET-$ENGINE.json, wall $(( ($(date +%s) - T0) / 60 )) min"
python3 -c "import json;r=json.load(open('$RES/$TARGET-$ENGINE.json'));h=r['hot_swap'];print(json.dumps({k:v for k,v in r.items() if k!='hot_swap'},indent=1));print(json.dumps({k:v for k,v in h.items() if k!='rows'},indent=1))"
