"""Pinned SDK-adapter tests; NumPy doubles are not actual Qwen model inference."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import numpy as np
from blocs.qwen_tts import logic, worker


class Model:
    def __init__(self, audio, tokens=None, rate=24000, call=True):
        self.audio, self.rate, self.call = audio, rate, call
        self.model = NS(talker=NS(generate=lambda **kw: NS(sequences=np.array([tokens if tokens is not None else [12, 13, 2150]]))))

    def generate_custom_voice(self, **kwargs):
        self.arguments = kwargs
        if self.call:
            self.model.talker.generate(eos_token_id=2150, max_new_tokens=kwargs["max_new_tokens"])
        return [self.audio], self.rate


class WorkerTests(unittest.TestCase):
    def test_defaults_and_all_numeric_bounds(self):
        package = Path(__file__).parents[1]
        fields = json.loads((package / "fields.json").read_text())
        self.assertEqual({s["key"] for s in fields}, set(logic.DEFAULTS))
        self.assertEqual(logic.DEFAULTS, json.loads((package / "model.json").read_text())["config"])
        for spec in fields:
            if spec["type"] in ("integer", "number"):
                for value in (True, None, "1", float("nan"), spec["min"] - 1, spec["max"] + 1):
                    with self.subTest(key=spec["key"], value=value), self.assertRaises(logic.QwenError):
                        logic.configuration({spec["key"]: value})
        for cfg in ({"speaker":"unknown"},{"language":"xx"},{"instruct":"x"*513},{"instruct":"\ud800"},{"memory_limit_mib":4095}):
            with self.subTest(cfg=cfg), self.assertRaises(logic.QwenError): logic.configuration(cfg)
        self.assertEqual(logic.configuration({"memory_limit_mib":4096})["memory_limit_mib"], 4096)

    def test_complete_audio_and_exact_arguments(self):
        cfg = logic.configuration({"language":"French", "speaker":"Aiden", "instruct":"Speak calmly"})
        model = Model(np.sin(np.arange(12000)/12) * 1.2)
        original = model.model.talker.generate
        with TemporaryDirectory() as directory:
            info = worker.generate_audio(model, "Bonjour", cfg, Path(directory))
            self.assertLess(info["gain"], 1)
            self.assertEqual(info["tokens"], 3)
            self.assertIs(model.model.talker.generate, original)
            self.assertEqual(logic.audio_info(Path(directory)/"generated.wav", cfg, json.dumps(info)), info)
        self.assertEqual(model.arguments, dict(text="Bonjour", speaker="Aiden", language="French", instruct="Speak calmly",
            non_streaming_mode=True, max_new_tokens=376, do_sample=True, temperature=.9, top_k=50,
            top_p=1., repetition_penalty=1.05, subtalker_dosample=True, subtalker_top_k=50,
            subtalker_top_p=1., subtalker_temperature=.9))

    def test_eos_missing_or_ambiguous_never_publishes(self):
        for tokens in ([1,2], [2150,2150], []):
            model = Model(np.array([.1]), tokens=tokens)
            original = model.model.talker.generate
            with TemporaryDirectory() as directory, self.subTest(tokens=tokens):
                with self.assertRaises(logic.QwenError):
                    worker.generate_audio(model, "Hello", logic.DEFAULTS, Path(directory))
                self.assertFalse(list(Path(directory).iterdir()))
                self.assertIs(model.model.talker.generate, original)
        with TemporaryDirectory() as directory, self.assertRaisesRegex(logic.QwenError, "end-of-sequence"):
            worker.generate_audio(Model(np.array([.1]), call=False), "Hello", logic.DEFAULTS, Path(directory))

    def test_invalid_audio_and_budget_are_refused(self):
        for audio in (np.array([]), np.array([0.]), np.array([[.2]]), np.array([np.nan]), np.array([np.inf]),
                      np.ones(24001), np.array([1], dtype=np.int16)):
            with TemporaryDirectory() as directory, self.subTest(shape=audio.shape):
                with self.assertRaises(logic.QwenError):
                    worker.generate_audio(Model(audio), "Hello", logic.configuration({"max_audio_sec":1}), Path(directory))
                self.assertFalse(list(Path(directory).iterdir()))
        with TemporaryDirectory() as directory:
            with self.assertRaises(logic.QwenError):
                worker.generate_audio(Model(np.array([.1]), rate=44100), "Hello", logic.DEFAULTS, Path(directory))
            model = Model(np.array([.1]))
            worker.generate_audio(model, "Hello", logic.configuration({"max_new_tokens":16}), Path(directory))
            self.assertEqual(model.arguments["max_new_tokens"], 16)

    def test_observer_rejects_unexpected_sdk_contract(self):
        for kwargs in ({"eos_token_id":3,"max_new_tokens":16}, {"eos_token_id":2150,"max_new_tokens":15}):
            talker = NS(generate=lambda **kw: NS(sequences=np.array([[1,2150]])))
            worker.guard_eos(talker, 16)
            with self.assertRaises(logic.QwenError): talker.generate(**kwargs)
        talker = NS(generate=lambda **kw: NS(sequences=np.array([[1,2150]])))
        _, state = worker.guard_eos(talker, 16)
        talker.generate(eos_token_id=2150,max_new_tokens=16)
        self.assertTrue(state["eos"])
        with self.assertRaises(logic.QwenError): talker.generate(eos_token_id=2150,max_new_tokens=16)

    def test_local_loader_has_no_remote_code_or_fallback(self):
        calls = []
        fake = NS(model=NS(tts_model_type="custom_voice", tts_model_size="1b7", tokenizer_type="qwen3_tts_tokenizer_12hz",
            config=NS(talker_config=NS(codec_eos_token_id=2150))),
            get_supported_speakers=lambda:["ryan"], get_supported_languages=lambda:["auto"])
        sdk = NS(Qwen3TTSModel=NS(from_pretrained=lambda *a,**kw:calls.append((a,kw)) or fake))
        with TemporaryDirectory() as directory, patch.dict(sys.modules,{"qwen_tts":sdk}):
            self.assertIs(worker.load_model(Path(directory),logic.DEFAULTS,NS(bfloat16="bf16")), fake)
            self.assertEqual(calls[0], ((directory,), dict(device_map="cuda:0", dtype="bf16", attn_implementation="sdpa",
                local_files_only=True, trust_remote_code=False)))
            fake.model.tts_model_type = "base"
            with self.assertRaises(logic.QwenError):worker.load_model(Path(directory),logic.DEFAULTS,NS(bfloat16="bf16"))


if __name__ == "__main__": unittest.main()
