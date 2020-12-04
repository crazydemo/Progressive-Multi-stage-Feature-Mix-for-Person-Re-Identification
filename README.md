# Progressive-Multi-stage-Feature-Mix-for-Person-Re-Identification

![alt text](https://github.com/crazydemo/Progressive-Multi-stage-Feature-Mix-for-Person-Re-Identification/blob/main/PMM_framework.png)


pytorch code for paper Progressive Multi-stage Feature Mix for Person Re-Identification: https://arxiv.org/abs/2007.08779

This project is based on batch-drop-block: https://github.com/daizuozhuo/batch-dropblock-network

The proposed PMM(Progressive-Multi-stage-Feature-Mix) model can be found in models/progressive_networks.py

## Setup running environment
This project requires python3, cython, torch, torchvision, scikit-learn, tensorboardX, fire.

## Prepare dataset
    
    Create a directory to store reid datasets under this repo via
    ```bash
    cd reid
    mkdir data
    ```
    
    For market1501 dataset, 
    1. Download Market1501 dataset to `data/` from http://www.liangzheng.org/Project/project_reid.html
    2. Extract dataset and rename to `market1501`. The data structure would like:
    ```
    market1501/
        bounding_box_test/
        bounding_box_train/
        query/
    ```

    For CUHK03 dataset,
    1. Download CUHK03-NP dataset from https://github.com/zhunzhong07/person-re-ranking/tree/master/CUHK03-NP 
    2. Extract dataset and rename folers inside it to cuhk-detect and cuhk-label.
    For DukeMTMC-reID dataset,
    Dowload from https://github.com/layumi/DukeMTMC-reID_evaluation
    
  ## Training

  ### Traning Market1501
```bash
python main_reid.py train --save_dir='./pytorch-ckpt/market-bfe' --max_epoch=400 --eval_step=30 --dataset=market1501 --test_batch=128 --train_batch=128 --optim=adam --adjust_lr
```
