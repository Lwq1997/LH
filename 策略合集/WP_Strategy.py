# -*- coding: utf-8 -*-
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


class WP_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)

    def select(self, context):
        log.info(self.name, '--Select函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        self.select_list = self.__get_rank(context)
        if self.max_industry_cnt > 0:
            self.select_list = self.utilstool.filter_stocks_by_industry(context, self.select_list,
                                                                        max_industry_stocks=self.max_industry_cnt)
        self.select_list = self.select_list[:self.max_select_count]

        if not self.select_list:
            self.select_list = [self.fill_stock]

        log.info(self.name, '的选股列表:', self.utilstool.getStockIndustry(self.select_list))
        self.print_trade_plan(context, self.select_list)

    def __get_rank(self, context):
        log.info(self.name, '--get_rank函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        initial_list = super().stockpool(context, 1, "399101.XSHE")

        q = query(
            valuation.code
        ).filter(
            valuation.code.in_(initial_list),
            # indicator.roa > 0,
            # indicator.adjusted_profit > 0,
            # income.np_parent_company_owners > 0,
            # income.net_profit > 0,
            # income.operating_revenue > 1e8
        ).order_by(
            valuation.market_cap.asc()  # 根据市值从小到大排序
        ).limit(self.max_select_count)

        final_stocks = list(get_fundamentals(q).code)
        return final_stocks
