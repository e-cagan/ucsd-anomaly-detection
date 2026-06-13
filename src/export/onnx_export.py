"""
Export the M3 U-Net predictor to ONNX, with a PyTorch-vs-ONNX parity check.
"""

import torch
import numpy as np
import onnxruntime as ort
from src.models.predictor import UNetPredictor


if __name__ == "__main__":
    device = "cpu"   # generally conducts on cpu

    # Load trained model
    model = UNetPredictor().to(device)
    ckpt = torch.load("checkpoints/pred_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Example input — same shape as a real window
    dummy = torch.randn(1, 15, 1, 128, 128)   # (B, 15, 1, H, W)

    # Export to ONNX
    torch.onnx.export(
        model,                          # model
        dummy,                          # sample input (for trace)
        "checkpoints/model.onnx",       # output filename
        input_names=["input"],          # input node name
        output_names=["output"],        # output node name
        dynamic_axes={                  # dynamic batch axis
            "input":  {0: "batch"},
            "output": {0: "batch"},
        },
        opset_version=18,
    )

    # Parity test — PyTorch vs ONNX Runtime
    with torch.no_grad():
        torch_out = model(dummy).cpu().numpy()

    sess = ort.InferenceSession("checkpoints/model.onnx")
    input_name = sess.get_inputs()[0].name
    onnx_out = sess.run(None, {input_name: dummy.numpy()})[0]

    # Compare model outputs
    max_diff = np.abs(torch_out - onnx_out).max()
    print(max_diff)  # expected ~1e-5
    assert max_diff < 1e-4