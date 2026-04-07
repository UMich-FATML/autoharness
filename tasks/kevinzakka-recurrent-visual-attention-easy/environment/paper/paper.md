# Recurrent Models of Visual Attention

Source: NIPS 2014 proceedings PDF, extracted from the public paper text because local PDF download was unavailable in the workspace shell.

## Abstract

Applying convolutional neural networks to large images is computationally expensive because the amount of computation scales linearly with the number of image pixels. We present a novel recurrent neural network model that is capable of extracting information from an image or video by adaptively selecting a sequence of regions or locations and only processing the selected regions at high resolution. Like convolutional neural networks, the proposed model has a degree of translation invariance built-in, but the amount of computation it performs can be controlled independently of the input image size. While the model is non-differentiable, it can be trained using reinforcement learning methods to learn task-specific policies.

## Section 3.1 Model

The agent is built around a recurrent neural network. At each time step it processes sensor data, integrates information over time, and chooses how to act and where to deploy its sensor next.

- Sensor: the bandwidth-limited sensor extracts a retina-like representation `rho(x_t, l_{t-1})` around location `l_{t-1}`. It encodes the region around `l` at high resolution and uses progressively lower resolution for pixels further away, producing a compact glimpse vector.
- Internal state: the hidden units `h_t` summarize the history of past observations and are updated by the core network `h_t = f_h(h_{t-1}, g_t; theta_h)`.
- Actions: the location action is chosen from a distribution parameterized by the location network `f_l(h_t; theta_l)`, while the task action comes from `f_a(h_t; theta_a)`.

## Section 3.2 Training

The paper optimizes expected reward with REINFORCE for the location policy and uses a learned baseline to reduce variance. For classification problems, the action network is also trained with cross-entropy on the known class label. The key hybrid objective is:

- use cross-entropy to train the action network and backpropagate through the core and glimpse networks
- always train the location network with REINFORCE

## Section 4 Design Choices

Retina and location encodings:

- the retina `rho(x, l)` extracts `k` square patches centered at location `l`
- the first patch is `g_w x g_w`
- each successive patch is twice as wide as the previous one
- all patches are resized back to `g_w x g_w` and concatenated
- locations are encoded as real-valued `(x, y)` coordinates with `(0, 0)` at the image center and `(-1, -1)` at the top-left corner

Glimpse network:

- `h_g = Rect(Linear(rho(x, l)))`
- `h_l = Rect(Linear(l))`
- `g = Rect(Linear(h_g) + Linear(h_l))`
- the paper uses dimensionality `128` for `h_g` and `h_l`, and `256` for `g`

Core network for classification:

- `h_t = Rect(Linear(h_{t-1}) + Linear(g_t))`

Action network:

- classification happens at the last timestep
- the action network is a linear softmax classifier over the final hidden state

Location network:

- the location policy is a two-component Gaussian with fixed variance
- the location network outputs the mean at time `t` as `f_l(h_t) = Linear(h_t)`

## Table 1

### 28x28 MNIST

| Model | Error |
|---|---:|
| FC, 2 layers (256 hiddens each) | 1.69% |
| Convolutional, 2 layers | 1.21% |
| RAM, 2 glimpses, 8x8, 1 scale | 3.79% |
| RAM, 3 glimpses, 8x8, 1 scale | 1.51% |
| RAM, 4 glimpses, 8x8, 1 scale | 1.54% |
| RAM, 5 glimpses, 8x8, 1 scale | 1.34% |
| RAM, 6 glimpses, 8x8, 1 scale | 1.12% |
| RAM, 7 glimpses, 8x8, 1 scale | 1.07% |

### 60x60 Translated MNIST

| Model | Error |
|---|---:|
| FC, 2 layers (64 hiddens each) | 6.42% |
| FC, 2 layers (256 hiddens each) | 2.63% |
| Convolutional, 2 layers | 1.62% |
| RAM, 4 glimpses, 12x12, 3 scales | 1.54% |
| RAM, 6 glimpses, 12x12, 3 scales | 1.22% |
| RAM, 8 glimpses, 12x12, 3 scales | 1.20% |

## Table 2

### 60x60 Cluttered Translated MNIST

| Model | Error |
|---|---:|
| FC, 2 layers (64 hiddens each) | 28.58% |
| FC, 2 layers (256 hiddens each) | 11.96% |
| Convolutional, 2 layers | 8.09% |
| RAM, 4 glimpses, 12x12, 3 scales | 4.96% |
| RAM, 6 glimpses, 12x12, 3 scales | 4.08% |
| RAM, 8 glimpses, 12x12, 3 scales | 4.04% |
| RAM, 8 random glimpses | 14.4% |

The paper explicitly notes that learned glimpse policies achieve much lower error than random glimpses, especially in cluttered settings.
