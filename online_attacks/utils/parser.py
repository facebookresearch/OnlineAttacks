# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from nestargs import NestedArgumentParser
from nestargs.parser import NestedNamespace as Namespace
from omegaconf import OmegaConf, DictConfig
from typing import Optional, Union


class ArgumentParser(NestedArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.configs = {}

    def add_config(self, name: str, config):
        if name in self.configs:
            raise KeyError()
        self.configs[name] = config
        config = OmegaConf.structured(config)
        self.add_argument_from_config(config, parent=name)

    def add_argument_from_config(
        self, config: DictConfig, parent: Optional[str] = None
    ):
        name = ""
        if parent is not None:
            name = parent + "."
        for key in config:
            if not OmegaConf.is_missing(config, key):
                if OmegaConf.is_list(config[key]):
                    self.add_argument(
                        "--" + name + key,
                        default=config[key],
                        nargs="+",
                        type=type(config[key][0]),
                    )
                elif OmegaConf.is_config(config[key]):
                    self.add_argument_from_config(config[key], name + key)
                else:
                    self.add_argument(
                        "--" + name + key, default=config[key], type=type(config[key])
                    )
            else:
                self.add_argument(name + key)

    def parse_args(self, args=None):
        args = super().parse_args(args)
        args = ArgumentParser.to_dict(args)
        args = OmegaConf.create(args)
        for key, value in self.configs.items():
            config = OmegaConf.structured(value)
            args[key] = OmegaConf.merge(config, args[key])
        return Namespace(**args)

    @staticmethod
    def to_dict(args: Union[Namespace, dict]):
        if isinstance(args, Namespace):
            args = vars(args)
            for key, value in args.items():
                args[key] = ArgumentParser.to_dict(value)
        return args

