#-*- coding: utf-8 -*-
# 如果你的文件包含中文, 请在文件的第一行使用上面的语句指定你的文件编码

# 用到策略及数据相关API请加入下面的语句(如果要兼容研究使用可以使用 try except导入
from kuanke.user_space_api import *
from Strategy import Strategy
from jqdata import *
from kuanke.wizard import *
import numpy as np
import pandas as pd
import talib as tl
from jqlib.technical_analysis import *
import datetime as datet


# 外盘ETF轮动策略
class All_Day2_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)
        self.etf_pool = [
            "511010.XSHG",  # 国债ETF
            "518880.XSHG",  # 黄金ETF
            "513100.XSHG",  # 纳指100
            # 2020年之后成立(注意回测时间)
            "515080.XSHG",  # 红利ETF
            "159980.XSHE",  # 有色ETF
            "162411.XSHE",  # 华宝油气LOF
            "159985.XSHE",  # 豆粕ETF
        ]
        # 标的仓位占比
        self.rates = [0.4, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05]
        self.min_volume = 2000

    def adjust(self, context):
        log.info(self.name, '--adject函数（全天候定制）--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        subportfolio = context.subportfolios[self.subportfolio_index]

        # 计算每个 ETF 的目标价值
        targets = {
            etf: subportfolio.total_value * rate
            for etf, rate in zip(self.etf_pool, self.rates)
        }

        # 获取当前持仓
        current_positions = subportfolio.long_positions
        log.info(self.name, '的选股列表:', targets,'--当前持仓--',current_positions)

        # 计算最小交易单位的价值（假设一手是100股）
        min_trade_value = {etf: current_positions[etf].price * 100 for etf in self.etf_pool}

        if not current_positions:  # 如果没有持仓
            for etf, target in targets.items():  # 遍历ETF
                self.utilstool.open_position(context, etf, target)
        else:
            # 先卖出
            for etf, target in targets.items():
                    value = current_positions[etf].value
                    minV = min_trade_value[etf]
                    if value - target > self.min_volume and value - target >= minV:
                        log.info(f'全天候策略开始卖出{etf}，仓位{target}')
                        self.utilstool.open_position(context, etf, target)

            self.balance_subportfolios(context)

            # 再买入
            for etf, target in targets.items():
                if etf in current_positions:
                    value = current_positions[etf].value
                else:
                    value = 0
                minV = min_trade_value[etf]
                if target - value > self.min_volume and target - value >= minV and minV <= subportfolio.available_cash:
                    log.info(f'全天候策略开始买入{etf}，仓位{target}')
                    self.utilstool.open_position(context, etf, target)