# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from mab2rec.pipeline import train
from mab2rec.config import train_config

if __name__ == '__main__':
    args = train_config()
    train(**args)
