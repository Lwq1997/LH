#-*- coding: utf-8 -*-
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
class BMZH_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)
        self.market_temperature = "warm"

    def select(self, context):
        log.info(self.name, '--select函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 根据市场温度设置选股条件，选出股票
        self.select_list = self.__get_rank(context)[:self.max_select_count]
        # 编写操作计划
        self.print_trade_plan(context, self.select_list)

    def __get_rank(self, context):
        log.info(self.name, '--get_rank函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        initial_list = super().stockpool_index(context, "000300.XSHG")

        # 2.根据市场温度进行选股
        # ·如果市场温度为"cold",则筛选条件包括：
        #   市净率（PB ratio)大于0且小于1
        #   经营活动现金流入小计大于0
        #   扣除非经常损益后的净利润大于2.5亿
        #   营业收入大于10亿
        #   净利润大于2.5亿
        #   经营活动现金流入小计与扣除非经常损益后的净利润之比大于2.0
        #   净资产收益率（ROA)大于1.5
        #   净利润同比增长率大于-15%
        #   并且股票代码在初始列表中。
        # 查询结果按照ROA与市净率的比值降序排列
        # 并限制最多返回 self.max_select_count+1只股票。
        if self.market_temperature == "cold":
            q = query(
                valuation.code,
            ).filter(
                valuation.pb_ratio > 0,
                valuation.pb_ratio < 1,
                cash_flow.subtotal_operate_cash_inflow > 0,  # 经营活动现金流入小计
                indicator.adjusted_profit > 2.5e8,  # 扣除非经常损益后的净利润(元)                          #>=2.5亿
                income.operating_revenue > 10e8,  # 营业收入(元)                                          #>=10亿
                income.net_profit > 2.5e8,  # 净利润(元)                                            #>=2.5亿
                cash_flow.subtotal_operate_cash_inflow / indicator.adjusted_profit > 2.0,  # 经营活动现金流入小计/扣除非经常损益后的净利润(元)
                indicator.inc_return > 1.5,  # 净资产收益率(扣除非经常损益)(%)
                indicator.inc_net_profit_year_on_year > -15,  # 净利润同比增长率(%)
                valuation.code.in_(initial_list)
            ).order_by(
                (indicator.roa / valuation.pb_ratio).desc()
            ).limit(self.max_select_count + 1)

        # 如果市场温度为"warm"
        # 则筛选条件与"cold"类似，但经营活动现金流入小计与扣除非经常损益后的净利润之比大于1.0
        #   净资产收益率大于2.0
        #   净利润同比增长率大于0%。

        elif self.market_temperature == "warm":
            q = query(
                valuation.code,
            ).filter(
                valuation.pb_ratio > 0,
                valuation.pb_ratio < 1,
                cash_flow.subtotal_operate_cash_inflow > 0,
                indicator.adjusted_profit > 2.5e8,  # 扣除非经常损益后的净利润(元)                          #>=2.5亿
                income.operating_revenue > 10e8,  # 营业收入(元)                                          #>=10亿
                income.net_profit > 2.5e8,  # 净利润(元)                                            #>=2.5亿
                cash_flow.subtotal_operate_cash_inflow / indicator.adjusted_profit > 1.0,
                indicator.inc_return > 2.0,
                indicator.inc_net_profit_year_on_year > 0,
                valuation.code.in_(initial_list)
            ).order_by(
                (indicator.roa / valuation.pb_ratio).desc()
            ).limit(self.max_select_count + 1)
        # 如果市场温度为hot, 则筛选条件包括：
        #   市净率大于3,
        #   经营活动现金流入小计大于0
        #   扣除非经常损益后的净利润大于2.5亿
        #   营业收入大于10亿
        #   净利润大于2.5亿
        #   经营活动现金流入小计与除非经常损益后的净利润之比大于0.5
        #   净资产收益率大于3.0
        #   净利润同比增长率大于20%
        #   并且股票代码在初始列表中
        # 查询结果按照ROA降序排列
        # 并限制最多返回 self.max_select_count+1只股票。
        elif self.market_temperature == "hot":
            q = query(
                valuation.code,
            ).filter(
                valuation.pb_ratio > 3,
                cash_flow.subtotal_operate_cash_inflow > 0,
                indicator.adjusted_profit > 2.5e8,  # 扣除非经常损益后的净利润(元)                          #>=2.5亿
                income.operating_revenue > 10e8,  # 营业收入(元)                                          #>=10亿
                income.net_profit > 2.5e8,  # 净利润(元)                                            #>=2.5亿
                cash_flow.subtotal_operate_cash_inflow / indicator.adjusted_profit > 0.5,
                indicator.inc_return > 3.0,
                indicator.inc_net_profit_year_on_year > 20,
                valuation.code.in_(initial_list)
            ).order_by(
                indicator.roa.desc()
            ).limit(self.max_select_count + 1)

        # 得到选股列表
        # 3.执行查询并获取选股列表：使用 get_fundamentals 函数执行查询，并将查询结果转换为股票代码列表，然后返回这个列表。
        check_out_lists = list(get_fundamentals(q).code)
        return check_out_lists

    #  这个函数的目的是根据沪深300指数的历史收盘价数据来评估市场温度，并根据市场温度的不同状态设置一个临时变量temp的值。
    def Market_temperature(self, context):
        log.info(self.name, '--Market_temperature函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        # 获取数据：使用attribute_history函数获取沪深300指数过去220天的收盘价数据。
        index300 = attribute_history('000300.XSHG', 220, '1d', ('close'), df=False)['close']

        # 计算市场高度：通过计算最近5天收盘价的平均值与过去220天收盘价的最小值之差，再除以过去220天收盘价的最大值与最小值之差，得到市场高度（market_height)。
        market_height = (mean(index300[-5:]) - min(index300)) / (max(index300) - min(index300))

        # 判断市场温度：根据市场高度的值，将市场温度分为三种状态：
        # ·如果市场高度小于0.20, 则市场温度为"cold"。
        # ·如果市场高度大于0.90, 则市场温度为"hot"。
        # ·如果过去60天内的最高收盘价与最低收盘价之比大于1.20, 则市场温度为"warm"
        if market_height < 0.20:
            self.market_temperature = "cold"
        elif market_height > 0.90:
            self.market_temperature = "hot"
        elif max(index300[-60:]) / min(index300) > 1.20:
            self.market_temperature = "warm"

        # 设置临时变量：根据市场温度的不同状态，设置临时变量temp
        # ·如果市场温度为"cold",则temp被设置为200。
        # ·如果市场温度为"warm",则 temp被设置为300。
        # ·如果市场温度为"hot",则 temp被设置为400。
        if self.market_temperature == "cold":
            temp = 200
        elif self.market_temperature == "warm":
            temp = 300
        else:
            temp = 400

        if context.run_params.type != 'sim_trade':
            # 画图
            # record(temp=temp)
            pass
