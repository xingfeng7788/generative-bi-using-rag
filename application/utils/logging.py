# -*- coding: UTF-8 -*-
# @Project ：generative-bi-using-rag 
# @FileName ：logging.py
# @Author ：dingtianlu
# @Date ：2024/8/22 13:55
# @Function  :
import os
import logging

LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()

logger = None


def getLogger():
    global logger
    if logger is not None:
        return logger

    # 创建日志记录器
    logger = logging.getLogger('application')
    logger.propagate = False
    logger.setLevel(LOG_LEVEL)
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(LOG_LEVEL)

    # 设置日志格式
    log_format = '%(asctime)s-[%(filename)s-->line:%(lineno)d]-%(levelname)s:%(message)s'
    formatter = logging.Formatter(log_format)

    # 设置日志处理器格式
    console_handler.setFormatter(formatter)

    # 添加日志处理器
    logger.addHandler(console_handler)

    return logger