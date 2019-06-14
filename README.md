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

## caffe_optimise_quantize
change caffe_root to  your dir where installed caffe

    python3 merge_bn_scale_droupout.py --model deploy.prototxt --weights yolov3.caffemodel --output_model deploy_mergebn.prototxt --output_weights yolov3_mergebn.caffemodel
    python3 caffe-int8-convert-tool-dev-weight.py --proto=deploy_mergebn.prototxt --model=yolov3_mergebn.caffemodel --mean 127.5 127.5 127.5 --norm=0.007843 --images=./testimgs/ --output=yolov3.table
    caffe2ncnn deploy_mergebn.prototxt yolov3_mergebn.caffemodel yolov3_int8.param yolov3_int8.bin 256 yolov3.table
    
then use the .param and .bin as usual,ncnn can rec int8 mode auto.
