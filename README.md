# TensorFlow Library:

## checkpoint_explore
explore the checkpoint's parameter's name and shape
### useage:
    python checkpoint_explore.py --model_checkpoint *.ckpt-*
    
## tensorboard_demo
a demo show how to add scalar and graph to tensorboard

## finetune_with_multi_models
if you add some layer to one trained model ,and you want to load the trained model's para and only want to
train the other parts of the model,you need this,include 2 functions,one for load trained model ,one for only
train the the other parts of the whole model