import os

import numpy as np
import onnx

import tvm
from tvm import autotvm
from tvm import relay
import tvm.relay.testing
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
from tvm.contrib.util import tempdir
import tvm.contrib.graph_runtime as runtime

import run_onnx_bert


import logging
logging.getLogger('autotvm').setLevel(logging.DEBUG)


def get_network(model_path, batch_size, input_dict_list, shape_dict_list):
    model = onnx.load(model_path)
    mod, params = relay.frontend.from_onnx(model, shape_dict_list[0])

    # TODO
    input_shape = (batch_size, 3, 224, 224)
    output_shape = (batch_size, 1000)
    return mod, params, input_shape, output_shape


#### DEVICE CONFIG ####
target = tvm.target.cuda()

#### TUNING OPTION ####
network = 'tvm-bert-gpu'
log_file = "%s.log" % network
dtype = 'float32'

tuning_option = {
    'log_filename': log_file,

    'tuner': 'xgb',
    #'n_trial': 2000,
    #'early_stopping': 600,
    'n_trial': 200,
    'early_stopping': 60,

    'measure_option': autotvm.measure_option(
        builder=autotvm.LocalBuilder(timeout=10),
        #runner=autotvm.LocalRunner(number=20, repeat=3, timeout=4, min_repeat_ms=150),
        #runner=autotvm.LocalRunner(number=2, repeat=3, timeout=4, min_repeat_ms=150),
        runner=autotvm.RPCRunner(
            'v100',  # change the device key to your key
            '0.0.0.0', 9190,
            number=20, repeat=3, timeout=4, min_repeat_ms=150)
    ),
}

# You can skip the implementation of this function for this tutorial.
def tune_tasks(tasks,
               measure_option,
               tuner='xgb',
               n_trial=1000,
               early_stopping=None,
               log_filename='tuning.log',
               use_transfer_learning=True,
               try_winograd=True):
    #if try_winograd:
    #    for i in range(len(tasks)):
    #        try:  # try winograd template
    #            tsk = autotvm.task.create(tasks[i].name, tasks[i].args,
    #                                      tasks[i].target, tasks[i].target_host, 'winograd')
    #            input_channel = tsk.workload[1][1]
    #            if input_channel >= 64:
    #                tasks[i] = tsk
    #        except Exception:
    #            pass

    # create tmp log file
    tmp_log_file = log_filename + ".tmp"
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    print("xychu: len(tasks)", len(tasks))
    for i, tsk in enumerate(reversed(tasks)):
        prefix = "[Task %2d/%2d] " %(i+1, len(tasks))

        # create tuner
        if tuner == 'xgb' or tuner == 'xgb-rank':
            tuner_obj = XGBTuner(tsk, loss_type='rank')
        elif tuner == 'ga':
            tuner_obj = GATuner(tsk, pop_size=100)
        elif tuner == 'random':
            tuner_obj = RandomTuner(tsk)
        elif tuner == 'gridsearch':
            tuner_obj = GridSearchTuner(tsk)
        else:
            raise ValueError("Invalid tuner: " + tuner)

        if use_transfer_learning:
            if os.path.isfile(tmp_log_file):
                tuner_obj.load_history(autotvm.record.load_from_file(tmp_log_file))

        # do tuning
        n_trial = min(n_trial, len(tsk.config_space))
        print("xychu: n_trial", n_trial)
        tuner_obj.tune(n_trial=n_trial,
                       early_stopping=early_stopping,
                       measure_option=measure_option,
                       callbacks=[
                           autotvm.callback.progress_bar(n_trial, prefix=prefix),
                           autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    if os.path.exists(tmp_log_file):
        autotvm.record.pick_best(tmp_log_file, log_filename)
        os.remove(tmp_log_file)
    else:
        print("xychu: no tmp file")


def tune_and_evaluate(tuning_opt):
    input_dict_list, shape_dict_list, eval_examples, extra_data = run_onnx_bert.prepare_inputs()
    # extract workloads from relay program
    print("Extract tasks...")
    mod, params, input_shape, out_shape = get_network("bertsquad8.onnx", 1, input_dict_list, shape_dict_list)
    tasks = autotvm.task.extract_from_program(mod["main"], target=target,
                                              #params=params, ops=(relay.op.nn.conv2d,))
                                              params=params, ops=(relay.op.nn.dense,))

    # run tuning tasks
    print("Tuning...")
    tune_tasks(tasks, **tuning_opt)

    # compile kernels with history best records
    with autotvm.apply_history_best(log_file):
        print("Compile...")
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build_module.build(
                mod, target=target, params=params)

        # export library
        tmp = tempdir()
        filename = "net.tar"
        lib.export_library(tmp.relpath(filename))

        # load parameters
        ctx = tvm.context(str(target), 0)
        module = runtime.create(graph, lib, ctx)

        #data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype(dtype))
        #module.set_input('data', data_tvm)
        #module.set_input(**params)

        module.set_input(**params)
        module.set_input(**input_dict_list[0])

        # evaluate
        print("Evaluate inference time cost...")
        ftimer = module.module.time_evaluator("run", ctx, number=1, repeat=600)
        prof_res = np.array(ftimer().results) * 1000  # convert to millisecond
        print("Mean inference time (std dev): %.2f ms (%.2f ms)" %
              (np.mean(prof_res), np.std(prof_res)))

# We do not run the tuning in our webpage server since it takes too long.
# Uncomment the following line to run it by yourself.

tune_and_evaluate(tuning_option)
