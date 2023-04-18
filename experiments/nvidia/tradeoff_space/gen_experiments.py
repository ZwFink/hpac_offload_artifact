import click
import os
import itertools
import yaml
from dataclasses import dataclass
from typing import Union
import itertools
import copy


@dataclass
class ExperimentConfiguration:
    benchmark_name: str
    items_per_thread: int
    blocksize: int
    trials: int

@dataclass
class PerfoExperimentConfiguration(ExperimentConfiguration):
    technique: str
    skip: Union[float, int]

@dataclass
class TAFExperimentConfiguration(ExperimentConfiguration):
    prediction_size: int
    history_size: int
    taf_width: int
    threshold: int

class ConfigurationGenerator:
    def __init__(self, conf=None):
        if conf:
            self.conf = conf
        else:
            self.conf = None

    def get(self):
        output = []
        vals = list(self.conf.values())

        # For itertools product, we want everything
        # to be a list, not scalars
        for v in vals:
            if isinstance(v, list):
                output.append(v)
            else:
                output.append([v])

        return output

    def combine_products(self, args):
        return itertools.product(*args,
                                 *self.get()
                                 )
    def get_labels(self):
        return list(self.conf.keys())

    def get_product(self):
        return itertools.product(*self.get())

    # combine self with other, inserting other at the end of self in order
    # return a new ConfigurationGenerator object with the intersection
    def combine_right(self, other):
        new_conf = copy.deepcopy(self.conf)
        new_conf.update(other.conf)
        return ConfigurationGenerator(conf=new_conf)

    def combine_left(self, other):
        new_conf = copy.deepcopy(other.conf)
        new_conf.update(self.conf)
        return ConfigurationGenerator(conf=new_conf)

class PercentPerfoConfigurationGenerator(ConfigurationGenerator):
    def __init__(self, arguments):
        self.types = list(set(['ini', 'fini']) & set(arguments['perfo_types']))
        self.percent = arguments['perfo_percent']
        self.conf = {'technique': self.types,
                     'skip': self.percent,
                     'items_per_thread': [1]
                     }

class SkipPerfoConfigurationGenerator(ConfigurationGenerator):
    def __init__(self, arguments):
        self.types = list(set(['small', 'large']) & set(arguments['perfo_types']))
        self.skips = arguments['perfo_skip']
        self.conf = {'technique': self.types,
                     'skip': self.skips,
                     'items_per_thread': arguments['items_per_thread']
                     }

    def get(self):
        return self._get_iter_skips()

class iACTConfigurationGenerator(ConfigurationGenerator):
    def __init__(self, arguments):
        # assume Python >= 3.7, where dictionaries are
        # ordered by insertion
        self.conf = arguments

class TAFConfigurationGenerator(ConfigurationGenerator):
    def __init__(self, arguments):
        # assume Python >= 3.7, where dictionaries are
        # ordered by insertion
        self.conf = arguments


class BenchmarkConfigurationGenerator(ConfigurationGenerator):
    def __init__(self, name, tech_config : ConfigurationGenerator, regions: list[str],
                 overrides = None, tech_overrides = None
                 ):
        self.conf = {'benchmark': name,
                     'region': regions
                     }
        self.conf.update(tech_config.conf)
        if overrides:
            self.conf.update(overrides)
        if tech_overrides:
            self.conf.update(tech_overrides)


class GeneralConfigurationGenerator(ConfigurationGenerator):
    def __init__(self, approx_type, trials, items_per_thread, blocksize):
        self.conf = {'approx_type': approx_type,'trials': trials,
                     'items_per_thread': items_per_thread,
                     'blocksize': blocksize
                     }

@click.command()
@click.option('--input', help="Input file containing configuration")
@click.option('--output', help="Output file")
@click.option('--technique', help="Approximate technique to generate experiments for. Can be one of: perfo, iact, taf.")
def main(input, output, technique):
    in_file = open(input, 'r')
    cfg = yaml.load(in_file, yaml.Loader)
    benchmarks = cfg['benchmark']
    gen = cfg['gen']
    # combine the general configuration generator with the technique-specific one
    gen_config = GeneralConfigurationGenerator(technique, gen['trials'],
                                               gen['items_per_thread'],
                                               gen['blocksize']
                                               )
    combined_config = None
    if 'perfo' == technique:
        skip_perfo_cfg = SkipPerfoConfigurationGenerator(gen['perfo'])
        percent_perfo_cfg = PercentPerfoConfigurationGenerator(gen['perfo'])
        combined_cfg_skip = gen_config.combine_right(skip_perfo_cfg)
        combined_cfg_percent = gen_config.combine_right(percent_perfo_cfg)
    elif 'iact' == technique:
        iact_cfg = iACTConfigurationGenerator(gen['iact'])
        combined_config = gen_config.combine_right(iact_cfg)
    elif 'taf' == technique:
        taf_cfg = TAFConfigurationGenerator(gen['taf'])
        combined_config = gen_config.combine_right(taf_cfg)
    else:
        raise ValueErorr(f"Unknown approximation technique: {technique}")

    configs = []

    for bench, config in benchmarks.items():
        overrides = None
        general_overrides = None
        technique_overrides = None
        if technique not in config['regions']:
            print(f"WARNING: No region found in benchmark {bench} for technique {technique}")
            continue
        if 'overrides' in config:
            overrides = config['overrides']
            general_overrides = {k: overrides[k] for k in overrides.keys() - {'perfo', 'iact', 'taf'}}
            if technique in config['overrides']:
                technique_overrides = overrides[technique]
        if 'perfo' == technique:
            bench_cfg = BenchmarkConfigurationGenerator(bench, combined_cfg_skip,
                                                        config['regions'][technique],
                                                        overrides=general_overrides,
                                                        tech_overrides=technique_overrides,
                                                        )
            configs.append(bench_cfg)
            bench_cfg = BenchmarkConfigurationGenerator(bench, combined_cfg_percent,
                                                        config['regions'][technique],
                                                        overrides=general_overrides,
                                                        tech_overrides=technique_overrides,
                                                        )
            configs.append(bench_cfg)
            overrides = None
        else:
            bench_cfg = BenchmarkConfigurationGenerator(bench, combined_config,
                                                        config['regions'][technique],
                                                        overrides=general_overrides,
                                                        tech_overrides=technique_overrides,
                                                        )
            configs.append(bench_cfg)
            overrides = None

    all_configs = []
    for b in configs:
        all_configs += list(b.get_product())

    label = ['experiment_number'] + configs[0].get_labels()
    header = ','.join(label)

    exp_num = 0
    with open(output, 'w') as out_file:
        out_file.write(f'{header}\n')
        for i, e in enumerate(all_configs):
            out_file.write(f'{",".join(map(str,[i+1]+list(e)))}\n')

if __name__ == '__main__':
    main()
