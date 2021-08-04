import json
import time
import sys
from abc import ABC, abstractmethod, ABCMeta
from datetime import datetime


class InfoSource(metaclass=ABCMeta):
    @abstractmethod
    def market_chart(self, target, start, period, end):
        pass
