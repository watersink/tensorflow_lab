import argparse
import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument("--model_checkpoint", help="specify checkpoint file", required=True)
args = parser.parse_args()

for var in tf.contrib.framework.list_variables(args.model_checkpoint):
    print(var[0])