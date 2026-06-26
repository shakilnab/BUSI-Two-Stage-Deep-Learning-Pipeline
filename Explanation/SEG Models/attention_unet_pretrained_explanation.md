# Attention U-Net Pretrained Explanation

## Code Context

Amader notebook-e Attention U-Net manually define kora hoyeche `class AttentionUNet(nn.Module)` er moddhe.

Main structure:

```text
Pretrained ResNet34 encoder
-> bottleneck
-> decoder with attention-gated skip connections
-> final 1-channel segmentation mask logits
```

Encoder code:

```python
if pretrained:
    resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
else:
    resnet = models.resnet34(weights=None)
```

Ei file-er focus:

```text
Attention U-Net pretrained=True
```

So encoder uses:

```text
ImageNet-1K V1 pretrained ResNet34 weights
```

---

## 1. Full Attention U-Net Architecture

```mermaid
flowchart LR
    A["Input<br/>RGB ultrasound image<br/>3 x 256 x 256"]

    B["enc1<br/>ResNet34 conv1 + BN + ReLU<br/>3 to 64<br/>64 x 128 x 128"]

    C["enc2<br/>MaxPool + ResNet layer1<br/>64 to 64<br/>64 x 64 x 64"]

    D["enc3<br/>ResNet layer2<br/>64 to 128<br/>128 x 32 x 32"]

    E["enc4<br/>ResNet layer3<br/>128 to 256<br/>256 x 16 x 16"]

    F["Bottleneck<br/>ResNet layer4<br/>256 to 512<br/>512 x 8 x 8"]

    G["dec4<br/>ConvTranspose 512 to 256<br/>AttentionGate with enc4<br/>Concat + ConvBlock 512 to 256<br/>256 x 16 x 16"]

    H["dec3<br/>ConvTranspose 256 to 128<br/>AttentionGate with enc3<br/>Concat + ConvBlock 256 to 128<br/>128 x 32 x 32"]

    I["dec2<br/>ConvTranspose 128 to 64<br/>AttentionGate with enc2<br/>Concat + ConvBlock 128 to 64<br/>64 x 64 x 64"]

    J["dec1<br/>ConvTranspose 64 to 64<br/>AttentionGate with enc1<br/>Concat + ConvBlock 128 to 64<br/>64 x 128 x 128"]

    K["up0 + final<br/>ConvTranspose 64 to 32<br/>Conv2d 1x1: 32 to 1<br/>Mask logits: 1 x 256 x 256"]

    L["Pretrained initialization<br/>pretrained=True<br/>Entire ResNet34 uses ImageNet-1K weights"]

    M["Every ConvBlock<br/>Conv3x3 + BN + ReLU<br/>Conv3x3 + BN + ReLU"]

    A --> B --> C --> D --> E --> F --> G --> H --> I --> J --> K

    L -. "applies to enc1 through bottleneck" .-> B
    L -.-> F

    M -.-> G
    M -.-> H
    M -.-> I
    M -.-> J

    E -. "att4 filtered skip" .-> G
    D -. "att3 filtered skip" .-> H
    C -. "att2 filtered skip" .-> I
    B -. "att1 filtered skip" .-> J

    classDef input fill:#e6f3ff,stroke:#0077cc,stroke-width:1.5px;
    classDef encoder fill:#dff1ff,stroke:#0095ff,stroke-width:1.5px;
    classDef bottleneck fill:#fff1cc,stroke:#f0a000,stroke-width:1.5px;
    classDef decoder fill:#d8fff5,stroke:#00a884,stroke-width:1.5px;
    classDef output fill:#e0ffe8,stroke:#1ab84f,stroke-width:1.5px;
    classDef note fill:#f5f7fa,stroke:#9aa4b2,stroke-width:1px;

    class A input;
    class B,C,D,E encoder;
    class F bottleneck;
    class G,H,I,J decoder;
    class K output;
    class L,M note;
```

Color meaning:

- Blue = pretrained ResNet34 encoder
- Yellow = bottleneck
- Green/teal = decoder with attention-gated skip connections
- Light green = output mask logits
- Gray = notes

---

## 2. Input

```text
Input
RGB ultrasound image
3 x 256 x 256
```

Model input holo RGB ultrasound image.

- `3` = RGB channels
- `256 x 256` = image height and width
- Model input hisebe only image jay
- Mask training-er target, model input na

Single image:

```text
3 x 256 x 256
```

Batch input:

```text
B x 3 x 256 x 256
```

Simple meaning:

**Attention U-Net image ney and corresponding tumour segmentation mask predict kore.**

---

## 3. Pretrained ResNet34 Encoder

Code:

```python
resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
```

Pretrained part:

```text
enc1
enc2
enc3
enc4
bottleneck
```

These all come from ResNet34.

Important:

```text
Only ResNet34 encoder/bottleneck uses pretrained ImageNet weights.
Decoder, AttentionGates, ConvBlocks, up0, final are trained for BUSI segmentation.
```

Presentation-safe line:

**Attention U-Net uses an ImageNet-1K pretrained ResNet34 encoder, while the attention gates and decoder are trained for the BUSI tumour segmentation task.**

---

## 4. Encoder and Bottleneck Count

Presentation-e evabe bolte paro:

```text
4 encoder stages
1 bottleneck
4 decoder stages
1 final output stage
```

Encoder:

```text
enc1, enc2, enc3, enc4
```

Bottleneck:

```text
ResNet layer4
```

Decoder:

```text
dec4, dec3, dec2, dec1
```

Final:

```text
up0 + final
```

---

## 5. enc1

```text
enc1
ResNet34 conv1 + BN + ReLU
3 to 64
64 x 128 x 128
```

Input:

```text
3 x 256 x 256
```

Output:

```text
64 x 128 x 128
```

Meaning:

- RGB image 64 feature map-e convert hoy
- spatial size `256 to 128`
- low-level features extract hoy: edge, texture, brightness pattern

---

## 6. enc2

```text
enc2
MaxPool + ResNet layer1
64 to 64
64 x 64 x 64
```

Input:

```text
64 x 128 x 128
```

Output:

```text
64 x 64 x 64
```

Meaning:

- MaxPool spatial size half kore
- ResNet layer1 feature refine kore
- channel same thake

---

## 7. enc3

```text
enc3
ResNet layer2
64 to 128
128 x 32 x 32
```

Input:

```text
64 x 64 x 64
```

Output:

```text
128 x 32 x 32
```

Meaning:

- spatial size `64 to 32`
- channel `64 to 128`
- deeper feature representation create hoy

---

## 8. enc4

```text
enc4
ResNet layer3
128 to 256
256 x 16 x 16
```

Input:

```text
128 x 32 x 32
```

Output:

```text
256 x 16 x 16
```

Meaning:

- spatial size `32 to 16`
- channel `128 to 256`
- higher-level tumour/tissue context capture hoy

---

## 9. Bottleneck

```text
Bottleneck
ResNet layer4
256 to 512
512 x 8 x 8
```

Input:

```text
256 x 16 x 16
```

Output:

```text
512 x 8 x 8
```

Meaning:

- encoder-er deepest feature representation
- spatial size most compressed
- semantic/context information highest

Simple meaning:

**Bottleneck image-er compact high-level summary create kore.**

---

## 10. AttentionGate Kibhabe Kaj Kore?

Code-er AttentionGate:

```python
g1 = self.W_g(g)
x1 = self.W_x(x)
psi = self.relu(g1 + x1)
psi = self.psi(psi)
return x * psi
```

Here:

- `g` = decoder gating signal
- `x` = encoder skip feature
- `psi` = attention map

Internal flow:

```mermaid
flowchart LR
    A["Decoder feature g"] --> B["W_g<br/>1x1 Conv + BN"]
    C["Encoder skip x"] --> D["W_x<br/>1x1 Conv + BN"]
    B --> E["Add"]
    D --> E
    E --> F["ReLU"]
    F --> G["psi<br/>1x1 Conv + BN + Sigmoid"]
    G --> H["Attention map"]
    C --> I["Multiply<br/>x * psi"]
    H --> I
    I --> J["Filtered skip feature"]

    classDef decoder fill:#d8fff5,stroke:#00a884,stroke-width:1.5px;
    classDef encoder fill:#dff1ff,stroke:#0095ff,stroke-width:1.5px;
    classDef att fill:#fff1cc,stroke:#f0a000,stroke-width:1.5px;
    classDef output fill:#e0ffe8,stroke:#1ab84f,stroke-width:1.5px;

    class A,B decoder;
    class C,D encoder;
    class E,F,G,H att;
    class I,J output;
```

Step-by-step:

1. Decoder feature `g` 1x1 Conv + BN diye process hoy.
2. Encoder skip feature `x` 1x1 Conv + BN diye process hoy.
3. Duita feature add hoy.
4. ReLU apply hoy.
5. `psi` path 1x1 Conv + BN + Sigmoid diye attention map create kore.
6. Encoder skip feature `x` attention map diye multiply hoy.
7. Output hoy filtered skip feature.

Simple meaning:

**AttentionGate encoder skip feature blindly pass kore na; decoder context use kore relevant tumour-region features emphasize kore.**

Presentation-safe line:

**The attention gate filters each encoder skip connection using the decoder signal, so the decoder receives more relevant spatial features for tumour segmentation.**

---

## 11. ConvBlock

Code-er ConvBlock:

```text
Conv3x3 + BN + ReLU
Conv3x3 + BN + ReLU
```

Purpose:

- concatenated decoder + filtered skip feature refine kora
- local boundary/detail improve kora
- segmentation feature clean kora

Simple meaning:

**Every decoder stage-e ConvBlock mixed feature-ke refine kore.**

---

## 12. Decoder Common Pattern

Each decoder stage follows:

```text
ConvTranspose
-> AttentionGate on encoder skip
-> Concat
-> ConvBlock
```

Meaning:

- ConvTranspose spatial size double kore
- AttentionGate skip feature filter kore
- Concat decoder and skip features combine kore
- ConvBlock combined feature refine kore

---

## 13. dec4

```text
dec4
ConvTranspose 512 to 256
AttentionGate with enc4
Concat + ConvBlock 512 to 256
256 x 16 x 16
```

Input:

```text
512 x 8 x 8
```

Upsample:

```text
512 x 8 x 8 -> 256 x 16 x 16
```

Skip:

```text
enc4 = 256 x 16 x 16
```

AttentionGate:

```text
enc4 filtered using dec4 signal
```

Concat:

```text
256 + 256 = 512 channels
```

ConvBlock:

```text
512 to 256
```

Output:

```text
256 x 16 x 16
```

---

## 14. dec3

```text
dec3
ConvTranspose 256 to 128
AttentionGate with enc3
Concat + ConvBlock 256 to 128
128 x 32 x 32
```

Input:

```text
256 x 16 x 16
```

Upsample:

```text
256 x 16 x 16 -> 128 x 32 x 32
```

Skip:

```text
enc3 = 128 x 32 x 32
```

Concat:

```text
128 + 128 = 256 channels
```

Output:

```text
128 x 32 x 32
```

---

## 15. dec2

```text
dec2
ConvTranspose 128 to 64
AttentionGate with enc2
Concat + ConvBlock 128 to 64
64 x 64 x 64
```

Input:

```text
128 x 32 x 32
```

Upsample:

```text
128 x 32 x 32 -> 64 x 64 x 64
```

Skip:

```text
enc2 = 64 x 64 x 64
```

Concat:

```text
64 + 64 = 128 channels
```

Output:

```text
64 x 64 x 64
```

---

## 16. dec1

```text
dec1
ConvTranspose 64 to 64
AttentionGate with enc1
Concat + ConvBlock 128 to 64
64 x 128 x 128
```

Input:

```text
64 x 64 x 64
```

Upsample:

```text
64 x 64 x 64 -> 64 x 128 x 128
```

Skip:

```text
enc1 = 64 x 128 x 128
```

Concat:

```text
64 + 64 = 128 channels
```

Output:

```text
64 x 128 x 128
```

---

## 17. up0 + final

```text
up0 + final
ConvTranspose 64 to 32
Conv2d 1x1: 32 to 1
Mask logits: 1 x 256 x 256
```

Input:

```text
64 x 128 x 128
```

up0:

```text
64 x 128 x 128 -> 32 x 256 x 256
```

final:

```text
32 x 256 x 256 -> 1 x 256 x 256
```

Output holo mask logits.

Important:

Logits probability na. Later sigmoid + threshold diye binary mask create hoy.

Simple meaning:

**Final stage original image size-e per-pixel tumour/background score output kore.**

---

## 18. Pretrained vs From Scratch

Attention U-Net pretrained and Attention U-Net from scratch er architecture same.

Difference:

```text
Pretrained Attention U-Net:
ResNet34 weights = ImageNet-1K V1

From-scratch Attention U-Net:
ResNet34 weights = None
```

So diagram mostly same. Sudhu initialization note different.

Presentation-safe line:

**The pretrained and from-scratch Attention U-Net models use the same architecture; the difference is encoder initialization.**

---

## 19. Why Attention U-Net?

Attention U-Net useful because:

- U-Net style encoder-decoder structure pixel-level segmentation-e strong
- skip connections high-resolution detail recover kore
- attention gates irrelevant skip features suppress kore
- tumour-relevant regions focus korte help kore

Presentation-safe line:

**Attention U-Net improves standard U-Net by filtering skip connections with attention gates, helping the decoder focus on tumour-relevant spatial features.**

---

## 20. Full Speaking Script

For Attention U-Net, amader code ImageNet-1K pretrained ResNet34 encoder use kore. Input holo 3-channel 256 by 256 ultrasound image. Encoder-e 4 ta stage ache: enc1, enc2, enc3, and enc4. Ei stages feature map-ke gradually downsample kore 64 by 128 by 128 theke 256 by 16 by 16 porjonto niye jay. Tarpor bottleneck ResNet layer4 output kore 512 by 8 by 8 high-level feature representation. Decoder side-e ConvTranspose use kore feature map gradually upsample hoy. Prottek decoder stage-e corresponding encoder skip feature AttentionGate diye filter hoy. AttentionGate decoder feature and encoder skip feature combine kore attention map banay, then skip feature-ke multiply kore relevant spatial feature pass kore. Tarpor decoder feature and filtered skip feature concatenate hoy and ConvBlock diye refine hoy. Finally up0 feature map-ke original 256 by 256 resolution-e niye ashe, and final 1x1 convolution 1-channel mask logits output kore for tumour segmentation.

---

## 21. Short Presentation Points

- Attention U-Net is a CNN-based segmentation model.
- Encoder: ImageNet-1K pretrained ResNet34.
- Input: `3 x 256 x 256` RGB ultrasound image.
- Structure: `4 encoder stages + 1 bottleneck + 4 decoder stages + final output`.
- Bottleneck output: `512 x 8 x 8`.
- Decoder uses ConvTranspose upsampling.
- Skip connections are filtered by AttentionGates.
- AttentionGate uses decoder signal to filter encoder skip feature.
- Every decoder stage uses ConvBlock.
- ConvBlock: `Conv3x3 + BN + ReLU`, repeated twice.
- Final output: `1 x 256 x 256` mask logits.
- Pretrained vs from scratch difference: encoder weight initialization only.

