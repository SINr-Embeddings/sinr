# -*- coding: utf-8 -*-

import logging

logger = logging.getLogger(__name__)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)

if not len(logger.handlers):
        logger.addHandler(ch)

logger.setLevel(logging.INFO)
