# Semantic Segmentation With Multi Scale Spatial Attention
Code for the paper `Semantic Segmentation With Multi Scale Spatial Attention For Self Driving Cars`.


Link to [Paper](https://abhinavsagar.github.io/files/sem_seg.pdf).

## Data

The dataset can be downloaded from [here](http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/).

## Network Architecture

![roc-auc](images/img1.png)

## Usage

Training: `python train.py`

Testing: `python test.py`

Testing on a single image: `python predict.py`

## Results

### Loss/Accuracy vs epochs

![roc-auc](images/img2.png)

### Comparision with SOTA

![roc-auc](images/img3.png)

### Results for predicted image

![roc-auc](images/img4.png)

## Citing

If you find this code useful in your research, please consider citing the paper:

```
@article{sagarsemantic,
  title={Semantic Segmentation With Multi Scale Spatial Attention For Self Driving Cars},
  author={Sagar, Abhinav and Soundrapandiyan, RajKumar}
}
```

## License

```
MIT License

Copyright (c) 2020 Abhinav Sagar

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

