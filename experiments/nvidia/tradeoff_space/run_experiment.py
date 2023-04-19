import click
import os
import sh
import pandas as pd
import yaml
from types import SimpleNamespace
from dataclasses import dataclass
import tempfile
import sqlite3
import struct
import numpy as np
import sys
import time
import glob
import io
import re
from pathlib import Path
from hpac_harness import *


@click.command()
@click.option('--technique', help="Approximation technique to use", type=click.Choice(['taf', 'iact', 'perfo']))
@click.option('--config_file', help="Name of the config YAML file", type=click.File('r'))
@click.option('--number', help="Experiment number to run", type=int)
def main(technique, config_file, number):
    cfg = yaml.load(config_file, yaml.Loader)
    run_cfg = cfg['run']
    bench_cfg = cfg['benchmark']
    exp_cfg_name = f'config_{technique}'
    exp_cfg = pd.read_csv(run_cfg[exp_cfg_name])
    exp_cfg = exp_cfg.set_index(exp_cfg['experiment_number']).loc[number]
    exp_cfg = exp_cfg.to_dict()
    exp_cfg.update(run_cfg)
    cfg.update(exp_cfg)
    cfg.update(run_cfg)
    cfg.update(bench_cfg)
    global_cfg = SimpleNamespace(**cfg)
    benchmark_name = exp_cfg['benchmark']

    hpac_build_dir = tempfile.TemporaryDirectory(prefix=run_cfg['temporary_base_dir'])
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    bench_cfg = modify_paths(bench_cfg, base_dir)
    cfg = modify_paths(cfg, base_dir)

    # Some approx techniques set environment for HPAC build

    approx_params = HPACApproxParams.create(exp_cfg['approx_type'], exp_cfg, bench_cfg[benchmark_name]['approx_arguments'])
    approx_build = approx_params.get_hpac_build_params()
    cfg.update(approx_build)
    build_hpac(hpac_build_dir.name, cfg)

    regions = []
    def apply_approx_regions():
        approx_tuple = approx_params.get_technique_tuple()

        src_dir = (Path(hpac_build_dir.name) / hpac_instance.get_benchmark_directory().stem).resolve()
        src_files = list(src_dir.glob('*.c')) + list(src_dir.glob('*.cpp')) + \
          list(src_dir.glob('*.C')) + list(src_dir.glob('*.cc')) + list(src_dir.glob('*.h')) + list(src_dir.glob('*.hpp'))

        _regions = find_approx_regions(src_dir, src_files)
        _regions = list(filter(lambda x: x.label == exp_cfg['region'], _regions))

        if _regions:
          regions.append(_regions[0])
          apply_approx_technique(src_dir, src_files, regions, approx_tuple, approx_params.get_technique_arg())
        #assert len(regions) > 0, f"ERROR: The region with label \'{exp_cfg['region']}\' was not found."
        #assert len(regions) == 1, f"ERROR: Multiple regions have the label: {exp_cfg['region']}"
        else:
          print("Warning: Running without approximation")
        print(_regions)

    print(f"Benchmark name is {benchmark_name}")
    bench_instance = HPACBenchmarkInstance.get_instance_class(benchmark_name)
    try:
        hpac_instance = bench_instance(benchmark_name,
                                       regions,
                                       bench_cfg[benchmark_name],
                                       hpac_build_dir.name
                                       )
        hpac_instance.build(pre=apply_approx_regions)
    except:
        regions.clear()
        # if it fails, we have one trick up our sleeve: try to allocate the table in global memory
        build_hpac(hpac_build_dir.name, cfg, _others = {'GLOBAL_OUTTABLE': '1'})
        hpac_instance = bench_instance(benchmark_name,
                                       regions,
                                       bench_cfg[benchmark_name],
                                       hpac_build_dir.name
                                       )
        hpac_instance.build(pre=apply_approx_regions)


    run_cmd = hpac_instance.get_run_command()

    #nblocks = hpac_instance.get_n() // exp_cfg['items_per_thread']
    if benchmark_name == 'leukocyte':
      nblocks = bench_cfg[benchmark_name]['executable_arguments']['num_cells']       
    elif benchmark_name == 'lavaMD':
        nblocks = 262144
    elif benchmark_name == 'binomialoptions':
      nblocks =  hpac_instance.get_n() // exp_cfg['items_per_thread']
    else:
      nblocks = HPACRuntimeEnvironment.calc_num_blocks(exp_cfg['items_per_thread'],
                                                               exp_cfg['blocksize'],
                                                               hpac_instance.get_n()
                                                               )

    rte = HPACRuntimeEnvironment(exp_cfg['blocksize'], nblocks, cfg['num_cpu_threads'])
    rte.configure_environment(exp_cfg['items_per_thread'],
                              exp_cfg['blocksize'],
                              nblocks
                               )

    experiment = HPACNodeExperiment(number, hpac_instance, rte, approx_params, None, 8)
    experiment.run_trials(global_cfg.trials)
    conn = sqlite3.connect(run_cfg['database_file'])
    experiment.write_info_to_db(conn, exp_cfg['approx_type'])
    conn.close()

def build_hpac(destination, cfg, _others = None):
    if not _others:
        _others = dict()
    def v_or_z(k):
        if k in cfg:
            return cfg[k]
        return 0

    _others['MAX_HIST_SIZE'] = v_or_z('MAX_HIST_SIZE')
    installer = HPACInstaller(cfg['hpac_basedir'], destination, f'{cfg["hpac_basedir"]}/clang', '15.0.0', v_or_z('enable_shared'),
                              v_or_z('DEV_STATS'), v_or_z('SHARED_MEMORY_SIZE'),
                              v_or_z('TABLES_PER_WARP'), v_or_z('TAF_WIDTH'),
                              other_options = _others
                              )

    installer.install()

def modify_paths(value, base_dir):
    if isinstance(value, str):
        path = Path(value)
        print(str(base_dir/ path))
        if not path.is_absolute() and Path(base_dir, path).exists():
            modified_path = Path(base_dir, path).resolve()
            if modified_path.exists():
                return str(modified_path)
        return value
    elif isinstance(value, dict):
        for key, val in value.items():
            value[key] = modify_paths(val, base_dir)
    return value

if __name__ == '__main__':
    main()
