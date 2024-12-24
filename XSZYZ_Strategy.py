# -*- coding: utf-8 -*-
# 如果你的文件包含中文, 请在文件的第一行使用上面的语句指定你的文件编码

# 用到策略及数据相关API请加入下面的语句(如果要兼容研究使用可以使用 try except导入
from kuanke.user_space_api import *
from Strategy import Strategy
from jqdata import *
from jqfactor import get_factor_values
import datetime
from kuanke.wizard import *
import numpy as np
import pandas as pd
import talib
from datetime import date as dt
import math
import talib as tl
from jqlib.technical_analysis import *
from scipy.linalg import inv
import pickle
import requests
import datetime as datet
from prettytable import PrettyTable
import inspect


# 白马股攻防转换策略（BMZH策略）
class XSZYZ_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)
        self.factor_list = params['factor_list']
        self.per_factor_max_select_count = params['per_factor_max_select_count']

    def select(self, context):
        log.info(self.name, '--select函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 根据市场温度设置选股条件，选出股票
        self.select_list = self.__get_rank(context)[:self.max_select_count]
        # 编写操作计划
        self.print_trade_plan(context, self.select_list)

    def __get_rank(self, context):
        log.info(self.name, '--get_rank函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()),'--当前策略因子:',self.factor_list)

        # TODO
        initial_list = super().stockpool(context)
        final_list = []

        for factor_list, coef_list in self.factor_list:
            factor_values = get_factor_values(initial_list, factor_list, end_date=context.previous_date, count=1)
            df = pd.DataFrame(index=initial_list, columns=factor_values.keys())
            for i in range(len(factor_list)):
                df[factor_list[i]] = list(factor_values[factor_list[i]].T.iloc[:, 0])
            df = df.dropna()
            df['total_score'] = 0
            for i in range(len(factor_list)):
                df['total_score'] += coef_list[i] * df[factor_list[i]]
            # 按照因子*因子比例计算总分
            df = df.sort_values(by=['total_score'], ascending=False)  # 分数越高即预测未来收益越高，排序默认降序
            complex_factor_list = list(df.index)[:int(0.1 * len(list(df.index)))]
            q = query(valuation.code, valuation.circulating_market_cap, indicator.eps).filter(
                valuation.code.in_(complex_factor_list)).order_by(valuation.circulating_market_cap.asc())
            df = get_fundamentals(q)
            df = df[df['eps'] > 0]
            lst = list(df.code)
            # lst = filter_paused_stock(lst)
            # lst = filter_limitup_stock(context, lst)
            # lst = filter_limitdown_stock(context, lst)
            lst = lst[:min(self.per_factor_max_select_count, len(lst))]
            log.error(self.name,'--factor_list:', factor_list, '选股列表:', lst)
            for stock in lst:
                if stock not in final_list:
                    final_list.append(stock)
        return final_list
