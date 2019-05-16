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

## pb_graph_explore
explore pb's parameter's name
### useage:
    python pb_graph_explore.py

## import_pb_to_tensorboard
load pb's graph to tensorboard,only use a .pb file
### useage:
    python import_pb_to_tensorboard.py --model_dir=./froze_graph.pb --log_dir=./log
    tensorboard --logdir=./log

## finetune_import_arbitrarily_op_with_tf
finetune with offical model,and changed class num,so the model shape is different,this code can help import arbitrarily op.
