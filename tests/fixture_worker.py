"""Synthetic native protocol fixture, NOT the Qwen model."""

import array
import json
import math
import os
from pathlib import Path
import sys
import time
import wave


def main():
    item=json.loads(sys.stdin.buffer.read(32769))
    assert os.environ.get("HF_HUB_OFFLINE")=="1"
    assert "QWEN_TEST_SENTINEL" not in os.environ
    root=Path(item["assets"]);request=item["request"];ident=request["request_id"]
    record=root/("started-"+ident+".json")
    temporary=root/("starting-"+ident)
    temporary.write_text(json.dumps({"pid":os.getpid(),"job":item["job"]}),encoding="utf-8")
    temporary.replace(record)
    if request["text"]=="fixture:hold":
        deadline=time.monotonic()+25
        while not (root/("release-"+ident)).exists() and time.monotonic()<deadline:
            time.sleep(.02)
    if request["text"]=="fixture:budget":
        print(json.dumps({"error_code":"audio_limit"}));return 1
    if request["text"]=="fixture:error":
        print("private third-party error",file=sys.stderr);print('{"error_code":"untrusted detail"}');return 1
    samples=array.array("h",(int(8000*math.sin(2*math.pi*440*n/24000)) for n in range(12000)))
    if sys.byteorder!="little":samples.byteswap()
    target=Path(item["job"])/"generated.wav"
    with wave.open(str(target),"wb") as output:
        output.setnchannels(1);output.setsampwidth(2);output.setframerate(24000);output.writeframes(samples.tobytes())
    info={"complete":True,"eos":True,"tokens":8,"sample_rate":24000,"frames":12000,"gain":1.}
    if request["text"]=="fixture:invalid":info["complete"]=False
    print(json.dumps(info));return 0


if __name__=="__main__":raise SystemExit(main())
