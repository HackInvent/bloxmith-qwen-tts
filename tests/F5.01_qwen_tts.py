"""FB1/FB2/FB3/FB4: adapter contracts and genuine processes/codecs; model is simulated."""

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
import threading
import time
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT),str(ROOT/"tests"),str(Path(__file__).parent)]
from blocs.qwen_tts import logic,worker
from qwen_fixture import SECRET,settings,alive,wait,started,check_file
from test_assets import AssetTests
from test_listener import ListenerTests


def refused(action,code):
    try:action()
    except logic.QwenError as exc:
        assert exc.code==code,(exc.code,code);return
    raise AssertionError("Invalid operation accepted: "+code)


from test_worker import WorkerTests


def parent_death(root):
    pidfile=root/"orphan-pid"
    code="\n".join([
        "import sys,time,types;from pathlib import Path",
        "p=types.ModuleType('owned');p.__path__=[sys.argv[1]];sys.modules[p.__name__]=p",
        "from owned.process import run_process",
        "child=\"import os,time;from pathlib import Path;Path(__import__('sys').argv[1]).write_text(str(os.getpid()));time.sleep(30)\"",
        "run_process([sys.executable,'-B','-c',child,sys.argv[2]],directory=Path(sys.argv[2]).parent,deadline=time.monotonic()+30,cancel=lambda:False,limit=1024)"])
    parent=subprocess.Popen([sys.executable,"-E","-B","-c",code,str(Path(logic.__file__).parent),str(pidfile)])
    child=None
    try:
        wait(pidfile.is_file);child=int(pidfile.read_text());parent.kill();parent.wait(timeout=3)
        wait(lambda:not alive(child),3)
    finally:
        if parent.poll() is None:parent.kill();parent.wait(timeout=3)
        if child and alive(child):os.kill(child,signal.SIGKILL)


def main():
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(case) for case in (AssetTests,ListenerTests,WorkerTests))
    result=unittest.TextTestRunner().run(suite);assert result.wasSuccessful()
    cfg=logic.configuration({})
    for value in ({"temperature":float("nan")},{"threads":True},{"max_pending":17},{"device":[]},{"unknown":"x"},{"memory_limit_mib":1024},{"asset_directory":"\ud800"}):
        refused(lambda:logic.configuration(value),"config")
    for value in (None,"",{"text":"x"},{"text":"x","request_id":"../x"},"\ud800","x"*1001):
        refused(lambda:logic.request(value,cfg),"input")
    assert logic.request('{"action":"interrupt"}',cfg)["text"]=='{"action":"interrupt"}'
    for value in ('{"action":"interrupt","action":"interrupt"}',{"action":"stop"},{"action":"interrupt","extra":1}):
        refused(lambda:logic.interrupt(value),"input")
    assert logic.interrupt('{"action":"interrupt"}')=={"action":"interrupt"}
    with TemporaryDirectory() as directory:
        root=Path(directory);storage=root/"storage";storage.mkdir();cfg=logic.configuration(settings(root))
        os.environ["QWEN_TEST_SENTINEL"]="private"
        try:
            # Execute the actual owned worker too: its dependency preflight must
            # fail cleanly on an interpreter without the separate Qwen environment.
            import importlib.util
            if importlib.util.find_spec("torch") is None:
                refused(lambda:logic.generate({"text":"Hello","request_id":"real-preflight"},
                    {**cfg,"environment_python":sys.executable},root,storage),"dependency")
            files=[]
            for fmt in ("wav","ogg_opus"):
                prepared=logic.generate({"text":"Hello","request_id":fmt},{**cfg,"format":fmt},root,storage)
                assert not (storage/"audio").exists() if not files else len(list((storage/"audio").iterdir()))==len(files)
                try:
                    metadata=logic.publish(prepared);files.append(check_file(metadata))
                finally:prepared.close()
            for text,code in (("fixture:invalid","audio_invalid"),("fixture:budget","audio_limit"),("fixture:error","model_failed")):
                refused(lambda:logic.generate({"text":text,"request_id":code},cfg,root,storage),code)
            cancelled=threading.Event();results=[]
            def job():
                try:logic.generate({"text":"fixture:hold","request_id":"cancel"},cfg,root,storage,cancelled.is_set)
                except Exception as exc:results.append(exc)
            thread=threading.Thread(target=job);thread.start();pid=started(cfg,"cancel")["pid"];cancelled.set();thread.join(5)
            assert not thread.is_alive() and results[0].code=="cancelled";wait(lambda:not alive(pid))
            refused(lambda:logic.generate({"text":"fixture:hold","request_id":"timeout"},{**cfg,"timeout_sec":1},root,storage),"timeout")
            prepared=logic.generate({"text":"Hello","request_id":"late"},cfg,root,storage)
            try:refused(lambda:logic.publish(prepared,lambda:True),"cancelled")
            finally:prepared.close()
            assert set((storage/"audio").iterdir())==set(files)
            assert not list(storage.glob("qwen-job-*")) and not list((storage/"audio").glob(".qwen*"))
            for command,limit,code in (("print('x'*10000)",10,"process_output"),("import time;time.sleep(3)",100,"timeout")):
                refused(lambda:logic.run_process([sys.executable,"-B","-c",command],directory=root,deadline=time.monotonic()+.4,cancel=lambda:False,limit=limit),code)
            parent_death(root)
        finally:os.environ.pop("QWEN_TEST_SENTINEL",None)
    print("[ok] Qwen adapter contracts, real process fencing/codecs/publication and asset/listener/SDK-observer tests; model inference simulated",flush=True)


if __name__=="__main__":main()
