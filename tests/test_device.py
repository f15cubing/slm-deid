"""Tests for backend/device detection (src/common/device.py).

These stay hardware-independent: they monkeypatch torch's availability probes so the
same assertions hold on a CUDA box (Colab) and an Apple-Silicon Mac.
"""

from __future__ import annotations

from src.common.device import detect_backend, pick_device


def test_explicit_backend_overrides_detection():
    assert detect_backend("hf") == "hf"
    assert detect_backend("unsloth") == "unsloth"


def test_auto_and_none_trigger_detection(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert detect_backend("auto") == "hf"
    assert detect_backend(None) == "hf"


def test_detect_backend_cuda_returns_unsloth(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert detect_backend() == "unsloth"


def test_detect_backend_without_cuda_returns_hf(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    assert detect_backend() == "hf"


def test_pick_device_prefers_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    assert pick_device() == "cuda"


def test_pick_device_falls_back_to_mps(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: True)
    assert pick_device() == "mps"


def test_pick_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("torch.backends.mps.is_available", lambda: False)
    assert pick_device() == "cpu"
