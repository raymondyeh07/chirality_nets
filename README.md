## Chirality Nets for Human Pose Regression
### NeurIPS 2019
<img src='./assets/intro.jpg' width=600>

#### [[Project]](http://chiralitynets.web.illinois.edu/) [[Paper]](https://arxiv.org/abs/1911.00029)

[Raymond A. Yeh](http://www.isle.illinois.edu/~yeh17/index.html)&ast;,
[Yuan-Ting Hu](https://sites.google.com/view/yuantinghu)&ast;, [Alexander G. Schwing](http://www.alexander-schwing.de/)<br/>
University of Illinois at Urbana-Champaign<br/>
(* indicates equal contribution)

The repository contains Pytorch implementation of Chirality Nets for Human Pose Regression.

If you used this code for your experiments or found it helpful, please consider citing the following paper:

<pre>
@inproceedings{YehNeurIPS2019,
  author = {R.~A. Yeh^\ast$ and Y.-T. Hu^\ast$ and A.~G. Schwing},
  title = {Chirality Nets for Human Pose Regression},
  booktitle = {Proc. NeurIPS},
  year = {2019},
  note = {$^\ast$ equal contribution},
}
</pre>

#### Dependencies:
* Python 3+
* Pytorch 1.1.0

### Usage
We recommend reading through our [short tutorial](./demo/equivariance_tutorial.html) on chirality equivariance. The tutorial illustrates the chirality definition and API for the chiral layers. The jupyter-notebook used to generate this tutorial.

#### Layers
We support chirality equivariant versions of the following layers:
* Linear
* Conv1D
* LSTM
* GRU
* Batch Normalization
To verify that these layers satisfies chirality equivariance, we have provided some test cases in ()

#### Applications
Coming soon.


### Related Work
* [3D human pose estimation in video with temporal convolutions and semi-supervised training](https://arxiv.org/abs/1811.11742) in CVPR 2019
* [Equivariance Through Parameter-Sharing](https://arxiv.org/abs/1702.08389) in ICML 2017

### License
This work is licensed under the MIT License
