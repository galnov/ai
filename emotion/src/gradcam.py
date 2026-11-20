"""
Grad-CAM - a visual explanation of the network's decision.

The idea: take the feature maps of the last convolution layer and check
(using the gradient) how much each map contributes to the score of the chosen emotion. A weighted
sum of the maps gives a "heat map" - the face regions that influenced the decision most.
"""
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        self._handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output
        if output.requires_grad:
            output.register_hook(self._save_grad)

    def _save_grad(self, grad):
        self.gradients = grad

    def __call__(self, x, class_idx=None):
        """
        x: a 1x1xHxW tensor.
        Returns (heatmap of size HxW in the range 0..1 as numpy, probabilities as numpy).
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
            if class_idx is None:
                class_idx = int(probs.argmax())
            logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)          # importance of each channel
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))  # weighted sum
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy(), probs.detach().cpu().numpy()

    def remove(self):
        self._handle.remove()
