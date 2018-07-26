#!/usr/bin/env python
# encoding: utf-8

# https://gist.github.com/nithishdivakar/c50696c5304555253b6a1a6aeff28d55
import tensorflow as tf
import os
import argparse
from argparse import RawTextHelpFormatter
from tensorflow.python.platform import gfile
from tensorflow.core.protobuf.meta_graph_pb2 import MetaGraphDef
from google.protobuf import text_format



def _parse_input_meta_graph_proto(input_graph, input_binary = True):
  """Parser input tensorflow graph into MetaGraphDef proto."""
  if not gfile.Exists(input_graph):
    print("Input meta graph file '" + input_graph + "' does not exist!")
    return -1
  input_meta_graph_def = MetaGraphDef()
  mode = "rb" if input_binary else "r"
  with gfile.FastGFile(input_graph, mode) as f:
    if input_binary:
      input_meta_graph_def.ParseFromString(f.read())
    else:
      text_format.Merge(f.read(), input_meta_graph_def)
  print("Loaded meta graph file '" + input_graph)
  return input_meta_graph_def

# TODO check argments
# reference python/tool/freeze_graph.py
def export_by_metagraph(checkpoint_dir, output_node_names, graph_file,
        graph_dir= ".", clear_devices = True, initializer_nodes = None,
        variable_names_whitelist = None, variable_names_blacklist = None,
        include_text = False):
    """
    Args:
        initializer_nodes : node names seperate by ","
    """
    if not output_node_names:
      print("You need to supply the name of a node to --output_node_names.")
      return -1
    ckpt = tf.train.latest_checkpoint(checkpoint_dir)
    if ckpt:
        input_meta_graph_def = _parse_input_meta_graph_proto(ckpt + ".meta")
        if clear_devices:
            for node in input_meta_graph_def.graph_def.node:
                node.device = ""
        with tf.Session() as sess:
            # restore all variables from checkpoint
            saver = tf.train.import_meta_graph(input_meta_graph_def, clear_devices = clear_devices)
            if saver is None:
                print("import meta graph from {0} error".format(ckpt))
                return None
            saver.restore(sess, ckpt)
            if initializer_nodes:
              sess.run(initializer_nodes.replace(" ", "").split(","))

            variable_names_whitelist = (
                variable_names_whitelist.replace(" ", "").split(",")
                if variable_names_whitelist else None)
            variable_names_blacklist = (
                variable_names_blacklist.replace(" ", "").split(",")
                if variable_names_blacklist else None)

            output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess,
                input_meta_graph_def.graph_def, # tf.get_default_graph().as_graph_def(), #sess.graph_def
                output_node_names.replace(" ", "").split(","),
                variable_names_whitelist=variable_names_whitelist,
                variable_names_blacklist=variable_names_blacklist)

            output_graph_def = tf.graph_util.remove_training_nodes(output_graph_def)

            pb_file = graph_file if graph_file.endswith(".pb") else graph_file + ".pb"
            tf.train.write_graph(output_graph_def, graph_dir, pb_file, as_text=False)
            print("export graph file {} successfully".format(graph_file))

            if include_text:
                pb_file_txt = graph_file.replace(".pb", ".pbtxt") if graph_file.endswith(".pb") else graph_file + ".pbtxt"
                tf.train.write_graph(output_graph_def, graph_dir, pb_file_txt , as_text=True)

            return output_graph_def
    else:
        print("{0} is not a valid checkpoint".format(ckpt))
        return None

# 其中 pb_path 为 graph 文件
# input_map : 为  graph 的 placeholder 组成的 dict，具体需要包括哪些，由 output_elements 的依赖决定
# output_elements : 输出的 operation 名称，格式为 op_name:output_index
def inference(pb_path, input_map, output_elements):
    with tf.Session() as sess:
        with open(pb_path, 'rb') as graph:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(graph.read())
            output = tf.import_graph_def(graph_def, return_elements=output_elements)
            return sess.run(output)

def _args_parse():
    #parser = argparse.ArgumentParser(description=description, formatter_class =
    #        RawTextHelpFormatter)
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        dest="checkpoint_dir",
        required=True,
        default=None,
        help='The checkpoint dir to be restore')

    parser.add_argument(
        '--output_graph',
        type=str,
        dest='output_graph',
        required=True,
        default=None,
        help='Output graph name.')

    parser.add_argument(
        '--output_graph_dir',
        type=str,
        dest='output_graph_dir',
        default=".",
        help='Output graph name.')

    parser.add_argument(
        "--output_node_names",
        type=str,
        dest='output_node_names',
        required=True,
        default=None,
        help="The name of the output nodes, comma separated.")

    parser.add_argument(
        "--clear_devices",
        nargs="?",
        const=True,
        type="bool",
        default=True,
        help="Whether to remove device specifications.")

    parser.add_argument(
        "--include_text",
        nargs="?",
        const=True,
        type="bool",
        default=False,
        help="Whether to export graph file of text format.")

    parser.add_argument(
        "--initializer_nodes",
        type=str,
        dest='initializer_nodes',
        default=None,
        help="Comma separated list of initializer nodes to run before freezing.")

    parser.add_argument(
        "--variable_names_whitelist",
        type=str,
        dest='variable_names_whitelist',
        default=None,
        help="""\
        Comma separated list of variables to convert to constants. If specified, \
        only those variables will be converted to constants.\
        """)

    parser.add_argument(
        "--variable_names_blacklist",
        type=str,
        dest='variable_names_blacklist',
        default=None,
        help="""\
        Comma separated list of variables to skip converting to constants.\
        """)

    args, unparsed = parser.parse_known_args()
    return args

def main():
    args = _args_parse()
    export_by_metagraph(args.checkpoint_dir, args.output_node_names, args.output_graph,
            args.output_graph_dir, args.clear_devices, args.initializer_nodes,
            args.variable_names_whitelist, args.variable_names_blacklist,
            args.include_text)



if __name__ == '__main__':
    main()
