"""CPU LoRA forward/gradient/merge checks and blockwise uniform 4-bit storage model."""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from torch import nn


class LoRALinear(nn.Module):
    def __init__(self, weight, rank, alpha=1.0):
        super().__init__()
        if weight.ndim != 2 or not isinstance(rank, int) or not 1 <= rank <= min(weight.shape):
            raise ValueError("Expected matrix weight and valid positive rank")
        self.register_buffer("weight", weight.detach().clone())
        self.a = nn.Parameter(torch.randn(rank, weight.shape[1], dtype=weight.dtype)*0.02)
        self.b = nn.Parameter(torch.zeros(weight.shape[0], rank, dtype=weight.dtype))
        self.scale = alpha/rank

    def forward(self, x):
        return x @ self.weight.T + ((x @ self.a.T) @ self.b.T)*self.scale

    def merged_weight(self):
        return self.weight + self.scale*(self.b @ self.a)


def quantize_uniform4(weight, block_size=64):
    if weight.numel() == 0 or not weight.is_floating_point() or not torch.isfinite(weight).all():
        raise ValueError("Expected nonempty finite floating-point tensor")
    if not isinstance(block_size, int) or block_size < 1:
        raise ValueError("block_size must be positive")
    flat = weight.detach().flatten()
    blocks = (len(flat)+block_size-1)//block_size
    padded = torch.zeros(blocks*block_size, dtype=weight.dtype)
    padded[:len(flat)] = flat
    padded = padded.reshape(blocks, block_size)
    scales = padded.abs().amax(dim=1).to(torch.float32)
    safe = torch.where(scales > 0, scales, torch.ones_like(scales))
    # Uniform codebook of 16 points in [-1,1]; this is NOT NF4.
    codes = torch.round((padded / safe[:, None] + 1)*7.5).clamp(0, 15).to(torch.uint8)
    if codes.numel() % 2:
        codes = torch.cat((codes.flatten(), torch.zeros(1, dtype=torch.uint8)))
    else:
        codes = codes.flatten()
    packed = codes[::2] | (codes[1::2] << 4)
    return {"packed": packed, "scales": scales, "shape": tuple(weight.shape),
            "block_size": block_size, "numel": weight.numel(), "dtype": weight.dtype}


def dequantize_uniform4(record):
    packed = record["packed"]
    codes = torch.stack((packed & 15, packed >> 4), dim=1).flatten()
    blocks = len(record["scales"])
    codes = codes[:blocks*record["block_size"]].to(record["dtype"]).reshape(blocks, record["block_size"])
    result = (codes/7.5-1)*record["scales"][:, None]
    return result.flatten()[:record["numel"]].reshape(record["shape"])


def run():
    torch.manual_seed(35)
    weight = torch.randn(12, 16, dtype=torch.float64)
    x = torch.randn(7, 16, dtype=torch.float64)
    target = torch.randn(7, 12, dtype=torch.float64)
    model = LoRALinear(weight, rank=3, alpha=6)
    initial_error = (model(x)-x @ weight.T).abs().max().item()
    loss = nn.functional.mse_loss(model(x), target)
    loss.backward()
    initial_gradient_norms = {"a": model.a.grad.norm().item(), "b": model.b.grad.norm().item()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer.step()
    torch.testing.assert_close(model.weight, weight, atol=0, rtol=0)
    merged = x @ model.merged_weight().T
    torch.testing.assert_close(model(x), merged, atol=1e-12, rtol=1e-12)
    record = quantize_uniform4(weight, 64)
    dequantized = dequantize_uniform4(record)
    return {"base_parameters": weight.numel(), "trainable_adapter_parameters": sum(p.numel() for p in model.parameters()),
            "initial_forward_max_abs_error": initial_error, "initial_gradient_norms": initial_gradient_norms,
            "base_weight_unchanged": True, "merged_max_abs_error": (model(x)-merged).abs().max().item(),
            "uniform4": {"packed_bytes": record["packed"].numel(), "fp32_scale_bytes": record["scales"].numel()*4,
                         "max_abs_weight_error": (dequantized-weight).abs().max().item(),
                         "output_max_abs_error": (x @ dequantized.T-x @ weight.T).abs().max().item()},
            "scope": "One adapter SGD update and merge on CPU; uniform packed quantization with FP32 scales, not NF4, double quantization, paged optimizer or actual QLoRA training"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results/35_lora_quantization.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {"recorded_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "device": "cpu", "torch": torch.__version__, "gpu_status": "skipped_by_user_request", **run()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
