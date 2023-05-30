# Image Zero-shot Classifier

## What is the difference zero-shot and fine-tuning?
### Zero-shot
| Zero-shot text classification is a task in natural language processing where a model is trained on a set of labeled examples but is then able to classify new examples from previously unseen classes. - [Huggingface zeroshot task](https://huggingface.co/tasks/zero-shot-classification)

Zero Shot is the task of predicting a class that unseened by the model during training. Capacity to make inferences about data that the model hasn't been trained on.
e.g. A model trained on ImageNet dataset. If A model has a enough capacity of zero-shot, the model can predict a class of a image that is not in ImageNet dataset. e.g. Medical image, Cartoon image, etc...

### Fine-tuning
| Unfreeze a few of the top layers of a frozen model base and jointly train both the **newly-added classifier layers** and the last layers of the base model. This allows us to "fine-tune" the **higher-order feature representations** in the base model in order to make them more relevant for the specific task. - [TF Transfer learning and fine-tuning](https://www.tensorflow.org/tutorials/images/transfer_learning)


## Clip-zero shot classifier example
- https://github.com/openai/CLIP/blob/main/notebooks/Interacting_with_CLIP.ipynb

### Run the cosine sim example
- banana / duck / duck-banana / cat / mountain / cat_mountain
- color
    - mountain is very similar to cat_mountain. (0.27310574 0.2711532)
    - duck banana is more simliar to duck than banana. (banana 0.2574472  duck 0.28605896 duck-banana 0.33556378)
```
python clip_cosine_sim.py

Descriptions with particular emphasis on color. sim : 
 [[0.34560353 0.2055461  0.25469753 0.18011668 0.17023855 0.14059618]
 [0.18314694 0.30653983 0.28971452 0.16112731 0.14310572 0.16548835]
 [0.2574472  0.28605896 0.33556378 0.1469627  0.15248975 0.15160184]
 [0.1670331  0.15959084 0.15298878 0.32098305 0.17914176 0.2640079 ]
 [0.19512695 0.17078938 0.15314542 0.20926546 0.27310574 0.2711532 ]
 [0.17777793 0.18318442 0.1645328  0.2726864  0.2668176  0.3479321 ]]

Descriptions with particular emphasis on shape. sim : 
 [[0.32092553 0.2162869  0.23260534 0.19564179 0.19288999 0.18030109]
 [0.1823345  0.29056168 0.28606778 0.17046987 0.15869159 0.1786189 ]
 [0.27503508 0.26821306 0.32447708 0.13652645 0.14753121 0.13049965]
 [0.14198351 0.14526635 0.13688743 0.2916434  0.17727861 0.27576113]
 [0.1950922  0.16079028 0.16869996 0.19931972 0.26041633 0.24714422]
 [0.20155513 0.16608208 0.16387594 0.254335   0.27246496 0.31465584]]
```


## Reference
- [Huggingface zeroshot task](https://huggingface.co/tasks/zero-shot-classification)
- [TF Transfer learning and fine-tuning](https://www.tensorflow.org/tutorials/images/transfer_learning)