# Licence of the model weights

The code of this repository is under the BSD 3-Clause licence (`LICENSE`).
**The model weights in this directory are not.** They are released under
[Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/)
(CC BY-NC 4.0): you may use and share them for non-commercial purposes, with
attribution.

The reason is the training data. The weights were trained on video that
includes material released for non-commercial use only, among it Meta's VC-1K
set (CC BY-NC 4.0) and the UVG sequences. Weights trained on permissively
licensed video alone can be made with the recipe in `recipe/`: its README
lists the datasets and how to leave one out.

Each directory holds a contract (`model.json`) and the network it names:
`model.i8`, the 8-bit integer form the page runs on WebGPU, or `model.onnx`,
a float graph for ONNX Runtime.
