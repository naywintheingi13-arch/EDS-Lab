# Sources and attribution

## Dataset

Parkhi, Vedaldi, Zisserman and Jawahar (2012), *Cats and Dogs*, CVPR.
https://www.robots.ox.ac.uk/~vgg/data/pets/

Four included pet-image crops and derivatives are attributed to the Oxford-IIIT Pet Dataset and
original image owners. Names identify original images. Modifications: shorter-edge resize to 256,
center crop to 224×224, heatmap overlays, and (in the stress test) a red/blue corner marker.
Trimaps receive the same geometric transform, with nearest-neighbor interpolation.

The official website consulted in the original session states CC BY-SA 4.0
(https://creativecommons.org/licenses/by-sa/4.0/); the original archive README has research-only wording
and requires respecting original website terms. The original README is preserved with this package.
This teaching/research package does not claim independent clearance for commercial redistribution.
Image derivatives follow CC BY-SA 4.0 to the extent permitted by source rights. Copyright remains
with original image owners. Do not interpret the mask as a ground-truth explanation.

## Model

Torchvision ResNet-18, ImageNet1K V1:
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
https://download.pytorch.org/models/resnet18-f37072fd.pth
https://github.com/pytorch/vision (project and license)

Original ImageNet head is removed. A class-balanced logistic regression head is fitted on frozen
512-dimensional features. No pixel-level fine-tuning. Official weights SHA-256 is saved in summary.json.

## Methods

- Selvaraju et al., Grad-CAM: https://arxiv.org/abs/1610.02391
- Sundararajan et al., Integrated Gradients: https://arxiv.org/abs/1703.01365
- Adebayo et al., Sanity Checks: https://arxiv.org/abs/1810.03292
- Completeness explanation: https://captum.ai/docs/extension/integrated_gradients

## Course mapping

Supplied *Deep Learning Explainability*, Sein Minn, AIT: p.71 Grad-CAM, p.93 IG, pp.101–106 visual
plausibility, randomization, Clever Hans and class discriminativity; p.114 practitioner checklist.
The completed core covers a bounded subset. SmoothGrad, label-randomized retraining, adversarial
robustness, TCAV and attention methods are extensions, not claimed completed experiments.
Using IG does not make an application high-stakes-ready.
