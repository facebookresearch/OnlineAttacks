# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import uuid
import json
import os
from omegaconf import OmegaConf
from typing import Optional
import glob
import shutil


def config_exists(config):
    if os.path.exists(config.save_dir):
        list_paths = os.listdir(config.save_dir)
    else:
        return None

    for exp_id in list_paths:
        logger = Logger(save_dir=config.save_dir, exp_id=exp_id)
        other_config = logger.load_hparams()
        other_config = OmegaConf.merge(config, other_config)
        if config == other_config:
            print("Found existing config with id=%s" % logger.exp_id)
            return logger.exp_id
    return None


class Logger:
    def __init__(self, save_dir: str, exp_id: Optional[str] = None):
        self.exp_id = exp_id
        if self.exp_id is None:
            self.exp_id = str(uuid.uuid4())
        self.path = os.path.join(save_dir, self.exp_id)

    def save_hparams(self, hparams):
        os.makedirs(self.path, exist_ok=True)
        filename = os.path.join(self.path, "hparams.yaml")
        OmegaConf.save(config=hparams, f=filename)

    def save_record(self, record):
        record_id = str(uuid.uuid4())
        os.makedirs(os.path.join(self.path, "runs"), exist_ok=True)
        filename = os.path.join(self.path, "runs/%s.json" % (record_id))
        with open(filename, "w+") as f:
            json.dump(record, f, indent=4)
            f.flush()
        return record_id

    def save_eval_results(self, eval_results, model_name, record_name):
        path = os.path.join(self.path, "eval", model_name)
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, "%s.json" % record_name)
        print("Saving eval results under %s" % filename)
        with open(filename, "w+") as f:
            json.dump(eval_results, f, indent=4)
            f.flush()

    def save_hist(self, eval_results, model_name, record_name):
        path = os.path.join(self.path, "hist", model_name)
        os.makedirs(path, exist_ok=True)
        filename = os.path.join(path, "%s.json" % record_name)
        with open(filename, "w+") as f:
            json.dump(eval_results, f, indent=4)
            f.flush()

    def load_hparams(self):
        filename = os.path.join(self.path, "hparams.yaml")
        return OmegaConf.load(filename)

    def load_record(self, name):
        filename = os.path.join(self.path, "runs", name + ".json")
        with open(filename, "r") as f:
            record = json.load(f)
        return record

    def __len__(self):
        return len(self.list_all_records())

    def list_all_records(self):
        path = os.path.join(self.path, "runs")
        list_records = []
        if os.path.exists(path):
            list_records = os.listdir(path)
            list_records = [os.path.splitext(filename)[0] for filename in list_records]
        return list_records

    def list_all_eval_models(self):
        path = os.path.join(self.path, "eval", "*/*")
        return glob.glob(path)

    def list_all_hist(self):
        path = os.path.join(self.path, "hist", "*/*")
        return glob.glob(path)

    def reset(self):
        path = os.path.join(self.path, "runs")
        if os.path.exists(path):
            shutil.rmtree(path)
        path = os.path.join(self.path, "eval")
        if os.path.exists(path):
            shutil.rmtree(path)
        path = os.path.join(self.path, "hist")
        if os.path.exists(path):
            shutil.rmtree(path)

    @staticmethod
    def list_all_logger(path):
        list_exp_id = glob.glob(os.path.join(path, "*"))
        list_exp_id = [os.path.basename(path) for path in list_exp_id]
        return list_exp_id

    @staticmethod
    def list_all_eval(path):
        return glob.glob(os.path.join(path, "*"))

    @staticmethod
    def load_eval_results(path):
        with open(path, "r") as f:
            record = json.load(f)
        return record

    def check_eval_results_exist(self, model_name, record_name):
        filename = os.path.join(self.path, "eval", model_name, "%s.json" % record_name)
        if os.path.exists(filename):
            return True
        return False

    def check_hist_exist(self, model_name, record_name):
        filename = os.path.join(self.path, "hist", model_name, "%s.json" % record_name)
        if os.path.exists(filename):
            return True
        return False

    def check_eval_done(self, model_name):
        path = os.path.join(self.path, "eval", model_name)
        if os.path.exists(path):
            list_dir = os.listdir(path)
        else:
            return False

        if len(list_dir) == len(self.list_all_records()):
            return True
        return False

    def check_hist_done(self, model_name):
        path = os.path.join(self.path, "hist", model_name)
        if os.path.exists(path):
            list_dir = os.listdir(path)
        else:
            return False

        if len(list_dir) == len(self.list_all_records()):
            return True
        return False
