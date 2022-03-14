# -*- coding: utf-8 -*-
# Copyright FMR LLC <opensource@fidelity.com>
# SPDX-License-Identifier: Apache-2.0

from mab2rec.pipeline import score
from mab2rec.config import score_config

if __name__ == '__main__':
    args = score_config()
    score(**args)
