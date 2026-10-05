"""Deterministic listener races and service failures; no native model execution."""

import json
from pathlib import Path
from queue import Empty,Queue
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

from blocs.qwen_tts import block,logic
from qwen_fixture import wait


class ListenerTests(unittest.TestCase):
    def context(self,root,storage):
        self.stop=threading.Event();self.commands=Queue();self.results=[];self.failures=[]
        def receive(**kw):
            try:return NS(payload=self.commands.get_nowait())
            except Empty:return None
        return NS(config=logic.configuration({}),root_dir=root,stop_requested=self.stop.is_set,
            services={"get_block_storage_dir":storage,"resolve_secret":lambda _:"[1,2,3,4,5]"},
            receive_command=receive,emit_result=self.results.append,
            output_ports=[NS(id=i+1,name=name) for i,name in enumerate(("audio_file","metadata","status"))])

    def start(self,context):
        def listen():
            try:block.QwenTtsBlock().listen_runtime(context)
            except Exception as exc:self.failures.append(exc)
        self.thread=threading.Thread(target=listen);self.thread.start()

    def finish(self):
        self.stop.set();self.thread.join(3)
        self.assertFalse(self.thread.is_alive());self.assertFalse(self.failures,self.failures)

    def statuses(self):
        return [json.loads(o.value) for r in self.results for o in r.outputs if o.port_name=="status"]

    def test_storage_failure_is_recoverable(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);attempts=[]
            def storage():
                attempts.append(1)
                if len(attempts)==1:raise OSError("private location")
                return root
            context=self.context(root,storage)
            with patch.object(logic,"generate",side_effect=logic.QwenError("dependency")):
                self.start(context)
                try:
                    self.commands.put({"action":"synthesize","text":"Hello","request_id":"storage"})
                    wait(lambda:any(s.get("code")=="storage" and s.get("request_id")=="storage" for s in self.statuses()))
                    self.assertTrue(self.thread.is_alive())
                    self.commands.put({"action":"synthesize","text":"Hello","request_id":"dependency"})
                    wait(lambda:any(s.get("code")=="dependency" and s.get("request_id")=="dependency" for s in self.statuses()))
                    self.assertNotIn("private location",str(self.statuses()))
                finally:self.finish()

    def test_queued_interrupt_fences_finished_future(self):
        with TemporaryDirectory() as directory:
            root=Path(directory);context=self.context(root,lambda:root);closed=[]
            def generate(*args):
                self.commands.put({"action":"interrupt"})
                return NS(close=lambda:closed.append(True))
            with patch.object(logic,"generate",side_effect=generate),patch.object(logic,"publish") as publish:
                self.start(context)
                try:
                    self.commands.put({"action":"synthesize","text":"Hello","request_id":"late"})
                    wait(lambda:any(s.get("code")=="interrupted" and s.get("request_id")=="late" for s in self.statuses()))
                    publish.assert_not_called();self.assertTrue(closed)
                    self.assertFalse(any(o.port_name=="audio_file" for r in self.results for o in r.outputs))
                finally:self.finish()

    def test_cached_text_not_replayed_by_command(self):
        attributes={"text":NS(status="consumed",value="old speech",content_type="text/plain"),
            "command":NS(status="updated",value='{"action":"interrupt"}',content_type="application/json")}
        context=NS(config={},input_events=[],input_ports=[NS(id=2,name="command"),NS(id=1,name="text")],
            input_attribute=lambda name,*aliases:attributes[name],runtime_mode="centralized",
            output_ports=[NS(id=3,name="status")])
        with patch.object(logic,"generate") as generate:
            result=block.QwenTtsBlock().execute_runtime(context)
            generate.assert_not_called();self.assertEqual(json.loads(result.outputs[0].value)["code"],"idle")


if __name__=="__main__":unittest.main()
