import time
import tvm
from tvm import relay
import numpy as np
import onnx
import onnx.utils
from onnx import helper, shape_inference
import os
from tvm.contrib import graph_runtime as runtime
import onnxruntime as ort
import tokenization
from run_onnx_squad import *
import json
import requests
import zipfile


def download_file(filename, url):
    """
    Download an URL to a file
    """
    with open(filename, 'wb') as fout:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        # Write response data to file
        for block in response.iter_content(4096):
            fout.write(block)


def download_if_not_exists(filename, url):
    """
    Download a URL to a file if the file
    does not exist already.
    Returns
    -------
    True if the file was downloaded,
    False if it already existed
    """
    if not os.path.exists(filename):
        download_file(filename, url)
        return True
    return False

def generate_input():
    input_question = {
      "version": "1.4",
      "data": [
        {
          "paragraphs": [
            {
              "context": "In its early years, the new convention center failed to meet attendance and revenue expectations.[12] By 2002, many Silicon Valley businesses were choosing the much larger Moscone Center in San Francisco over the San Jose Convention Center due to the latter's limited space. A ballot measure to finance an expansion via a hotel tax failed to reach the required two-thirds majority to pass. In June 2005, Team San Jose built the South Hall, a $6.77 million, blue and white tent, adding 80,000 square feet (7,400 m2) of exhibit space",
              "qas": [
                {
                  "question": "where is the businesses choosing to go?",
                  "id": "1"
                },
                {
                  "question": "how may votes did the ballot measure need?",
                  "id": "2"
                },
                {
                  "question": "By what year many Silicon Valley businesses were choosing the Moscone Center?",
                  "id": "3"
                }
              ]
            }
          ],
          "title": "Conference Center"
        }
      ]
    }
    with open('inputs.json', 'w') as f:
        json.dump(input_question, f)


def get_bert_files():
    bert_squad_8_url = "https://github.com/onnx/models/raw/master/text/machine_comprehension/bert-squad/model/bertsquad8.onnx"
    bert_squad_10_url = "https://github.com/onnx/models/raw/master/text/machine_comprehension/bert-squad/model/bertsquad10.onnx"
    tokenizer_url = "https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip"

    download_if_not_exists("bertsquad8.onnx", bert_squad_8_url)
    download_if_not_exists("bertsquad10.onnx", bert_squad_10_url)

    if not os.path.isdir("uncased_L-12_H-768_A-12"):
        download_if_not_exists("uncased_L-12_H-768_A-12.zip", tokenizer_url)
        with zipfile.ZipFile("uncased_L-12_H-768_A-12.zip","r") as zip_ref:
            zip_ref.extractall(".")


def prepare_inputs():
    predict_file = 'inputs.json'

    # Use read_squad_examples method from run_onnx_squad to read the input file
    eval_examples = read_squad_examples(input_file=predict_file)

    max_seq_length = 256
    doc_stride = 128
    max_query_length = 64

    vocab_file = os.path.join(
        'uncased_L-12_H-768_A-12', 'vocab.txt')
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)

    # Use convert_examples_to_features method from run_onnx_squad to get parameters from the input
    input_ids, input_mask, segment_ids, extra_data = convert_examples_to_features(eval_examples, tokenizer,
                                                                                  max_seq_length, doc_stride, max_query_length)
    n = len(input_ids)
    print("input_ids:", n)
    input_dict_list = []
    shape_dict_list = []
    for idx in range(0, n):
        item = eval_examples[idx]
        # this is using batch_size=1
        # feed the input data as int64
        input_dict = {"unique_ids_raw_output___9:0": np.array([item.qas_id], dtype=np.int64),
                      "input_ids:0": input_ids[idx:idx+1],
                      "input_mask:0": input_mask[idx:idx+1],
                      "segment_ids:0": segment_ids[idx:idx+1]}
        input_dict_list.append(input_dict)
        shape_dict = {"unique_ids_raw_output___9:0": np.array([item.qas_id], dtype=np.int64).shape,
                      "input_ids:0": input_ids[idx:idx+1].shape,
                      "input_mask:0": input_mask[idx:idx+1].shape,
                      "segment_ids:0": segment_ids[idx:idx+1].shape}
        shape_dict_list.append(shape_dict)
    return input_dict_list, shape_dict_list, eval_examples, extra_data


def postprocess(outputs, output_dir, eval_examples, extra_data):
    # postprocessing
    os.makedirs(output_dir, exist_ok=True)
    output_prediction_file = os.path.join(output_dir, "predictions.json")
    output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")
    write_predictions(eval_examples, extra_data, outputs,
                      20, 30,
                      True, output_prediction_file, output_nbest_file)
    with open(output_prediction_file) as json_file:
        test_data = json.load(json_file)
        print(json.dumps(test_data, indent=2))


def run_onnx_model(model_path, input_dict_list):
    all_results = []
    onnx_rt = ort.InferenceSession(model_path)
    n = len(input_dict_list)
    #
    onnx_rt.run(["unique_ids:0","unstack:0", "unstack:1"], input_dict_list[0])
    #
    s = time.time()
    for idx in range(0, n):
        result = onnx_rt.run(["unique_ids:0","unstack:0", "unstack:1"], input_dict_list[idx])
        start_logits = [float(x) for x in result[1][0].flat]
        end_logits = [float(x) for x in result[2][0].flat]
        unique_id = len(all_results)
        all_results.append(RawResult(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits))
    print("onnx time cost:", time.time() - s)
    return all_results


def run_tvm_model(model_path, input_dict_list, shape_dict_list):
    model = onnx.load(model_path)
    mod, params = relay.frontend.from_onnx(model, shape_dict_list[0])
    with relay.build_config(opt_level=0):
        #graph, lib, params = relay.build(mod, target='llvm', params=params)
        graph, lib, params = relay.build(mod, target='llvm -mcpu=skylake', params=params)
    gmod = runtime.create(graph, lib, ctx=tvm.cpu())
    gmod.set_input(**params)
    n = len(input_dict_list)
    all_results = []
    #
    gmod.set_input(**input_dict_list[0])
    gmod.run()
    #
    s = time.time()
    for idx in range(0, n):
        gmod.set_input(**input_dict_list[idx])
        gmod.run()
        start_logits = [float(x) for x in gmod.get_output(1).asnumpy()[0].flat]
        end_logits = [float(x) for x in gmod.get_output(0).asnumpy()[0].flat]
        unique_id = len(all_results)
        all_results.append(RawResult(unique_id=unique_id, start_logits=start_logits, end_logits=end_logits))
    print("tvm time cost:", time.time() - s)
    return all_results


if __name__ == "__main__":
    generate_input()
    get_bert_files()
    input_dict_list, shape_dict_list, eval_examples, extra_data = prepare_inputs()
    print("input_dict_lists:", len(input_dict_list))
    onnx_results = run_onnx_model("bertsquad8.onnx", input_dict_list)
    tvm_results = run_tvm_model("bertsquad8.onnx", input_dict_list, shape_dict_list)
    postprocess(onnx_results, "onnx_predictions", eval_examples, extra_data)
    postprocess(tvm_results, "tvm_predictions", eval_examples, extra_data)
