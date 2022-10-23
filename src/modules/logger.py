"""
@Desc:
Rrinting log to both screen and files.
@Reference:
https://xnathan.com/2017/03/09/logging-output-to-screen-and-file/
"""

import logging
from logging import Logger
from logging import NOTSET
import os
import sys
from pathlib import Path

FILE_PATH = Path(__file__).absolute()
BASE_DIR = FILE_PATH.parent.parent.parent
sys.path.insert(0, str(BASE_DIR))  # 在tasks文件夹中可以直接运行程序

DEBUG = logging.DEBUG


class MyLogger(Logger):
    def __init__(self, name, log_path='', level=NOTSET, log_stream='both'):
        super(MyLogger, self).__init__(name, level)
        self.log_path = log_path if log_path else f'{BASE_DIR}/output/log/{name}_log.txt'
        self.log_stream = log_stream
        self._handlers = {}
        self._init()

    def _init(self):
        warns = ''

        self.setLevel(self.level)
        formatter = logging.Formatter(
            "%(asctime)s %(pathname)s %(filename)s %(funcName)s %(lineno)s %(levelname)s - %(message)s",
            "%Y-%m-%d %H:%M:%S")

        # 使用FileHandler输出到文件
        if not os.path.exists(self.log_path):
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            warns = f'{self.log_path} did not exist, and has been created now.'
        file_handler = logging.FileHandler(f'{self.log_path}', encoding='utf-8')
        file_handler.setLevel(self.level)
        file_handler.setFormatter(formatter)

        # 使用StreamHandler输出到屏幕
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(self.level)
        stream_handler.setFormatter(formatter)

        # 添加Handler
        if self.log_stream == 'file':
            self.addHandler(file_handler)
            self._handlers['file'] = file_handler
        elif self.log_stream == 'screen':
            self.addHandler(stream_handler)
            self._handlers['screen'] = stream_handler
        elif self.log_stream == 'both':
            self.addHandler(stream_handler)
            self.addHandler(file_handler)
            self._handlers['file'] = file_handler
            self._handlers['screen'] = stream_handler
        else:
            raise ValueError('log_stream could only be "file", "screen", or "both"')

        # output warning
        if warns:
            self.warning(warns)
        self.info(
            f'MyLogger instance {self.name} has been set. level: {self.level}, log_path: {os.path.abspath(self.log_path)}')

    def clear_screen_handler(self):
        if 'screen' in self._handlers and self._handlers['screen'] is not None:
            self.removeHandler(self._handlers['screen'])
            del self._handlers['screen']

    def clear_file_handler(self):
        if 'file' in self._handlers and self._handlers['file'] is not None:
            self.removeHandler(self._handlers['file'])
            del self._handlers['file']

    def clear_all_handlers(self):
        self.clear_screen_handler()
        self.clear_file_handler()
