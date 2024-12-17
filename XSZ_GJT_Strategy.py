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
from UtilsToolClass import UtilsToolClass


class XSZ_GJT_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)
        self.new_days = 375  # 400 # 已上市天数
        self.highest = 50

    def select(self, context):
        log.info(self.name, '--select选股函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 空仓期控制和止损期都不在选择股票
        if self.use_empty_month and context.current_dt.month in (self.empty_month):
            log.info('Select选股函数不再执行，因为当前月份是空仓期，空仓期月份为：', self.empty_month)
            return
        if self.stoplost_date is not None:
            log.info('Select选股函数不再执行，因为当前时刻还处于止损期，止损期从:', self.stoplost_date, '开始')
            return

        self.select_list = self.__get_rank(context)[:self.max_select_count]
        self.print_trade_plan(context, self.select_list)

    def __get_rank(self, context):
        log.info(self.name, '--get_rank函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 获得初始列表
        initial_list = self.stockpool(context, 1, '399101.XSHE')
        # 过滤次新股
        initial_list = self.utilstool.filter_new_stock(context, initial_list, self.new_days)
        # 过滤120天内即将大幅解禁
        initial_list = self.utilstool.filter_locked_shares(context, initial_list, 120)

        final_list_1 = []
        # 市值5-30亿，并且在列表中，按市值从小到大到排序
        q = (query(valuation.code, valuation.market_cap)
             .filter(valuation.code.in_(initial_list), valuation.market_cap.between(5, 30))
             .order_by(valuation.market_cap.asc()))

        # 获取财务数据
        df_fun = get_fundamentals(q)[:50]
        # log.info(self.name, '过滤停盘/涨停/跌停之后，--前50股票的财务数据:', df_fun)
        final_list_1 = list(df_fun.code)

        # 获得初始列表
        lists = self.stockpool(context, 1, '399101.XSHE')
        # 过滤次新股
        lists = self.utilstool.filter_new_stock(context, lists, self.new_days)
        # 过滤120天内即将大幅解禁
        lists = self.utilstool.filter_locked_shares(context, lists, 120)
        final_list_2 = []
        # 国九更新：过滤近一年净利润为负且营业收入小于1亿的
        # 国九更新：过滤近一年期末净资产为负的 (经查询没有为负数的，所以直接pass这条)
        # 国九更新：过滤近一年审计建议无法出具或者为负面建议的 (经过净利润等筛选，审计意见几乎不会存在异常)
        q = query(
            valuation.code,
            valuation.market_cap,  # 总市值 circulating_market_cap/market_cap
            income.np_parent_company_owners,  # 归属于母公司所有者的净利润
            income.net_profit,  # 净利润
            income.operating_revenue  # 营业收入
            # security_indicator.net_assets
        ).filter(
            valuation.code.in_(lists),
            valuation.market_cap.between(5, 30),
            income.np_parent_company_owners > 0,
            income.net_profit > 0,
            income.operating_revenue > 1e8
        ).order_by(valuation.market_cap.asc()).limit(50)

        df = get_fundamentals(q)

        final_list_2 = list(df.code)
        last_prices = history(1, unit='1d', field='close', security_list=final_list_2)
        # 过滤价格低于最高价50元/股的股票  ｜  再持仓列表中的股票
        final_list_2 = [stock for stock in final_list_2 if
                        stock in self.hold_list or last_prices[stock][-1] <= self.highest]

        # 合并两个股票列表并去重
        target_list = list(dict.fromkeys(final_list_1 + final_list_2))
        # 取前 self.max_select_count * 3 只股票
        target_list = target_list[:self.max_select_count * 3]
        final_list = get_fundamentals(query(
            valuation.code,
            indicator.roe,
            indicator.roa,
        ).filter(
            valuation.code.in_(target_list),
            # valuation.pb_ratio<1
        ).order_by(
            valuation.market_cap.asc()
        )).set_index('code').index.tolist()
        return final_list
