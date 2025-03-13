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


class PJ_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)
        self.fill_stock = "518880.XSHG"

    def select(self, context):
        log.info(self.name, '--Select函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        self.select_list = self.__get_rank(context)
        if self.max_industry_cnt > 0:
            self.select_list = self.utilstool.filter_stocks_by_industry(context, self.select_list,
                                                                        max_industry_stocks=self.max_industry_cnt)
        self.select_list = self.select_list[:self.max_select_count]

        if not self.select_list:
            self.select_list = [self.fill_stock]

        log.info(self.name, '的选股列表:', self.select_list)
        self.print_trade_plan(context, self.select_list)

    def __get_rank(self, context):
        log.info(self.name, '--get_rank函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 获取股票池
        stocks = self.stockpool(context, all_filter=True)
        # q = query(
        #     valuation.code, valuation.market_cap, valuation.pe_ratio, income.total_operating_revenue
        # ).filter(
        #     # valuation.pb_ratio < 1,  # 破净
        #     # cash_flow.subtotal_operate_cash_inflow > 1e6,  # 经营现金流
        #     # indicator.adjusted_profit > 1e6,  # 扣非净利润
        #     # indicator.roa > 0.15,  # 总资产收益率
        #     # indicator.inc_operation_profit_year_on_year > 0,  # 净利润同比增长
        #
        #     valuation.pb_ratio > 0,
        #     valuation.pb_ratio < 1,
        #     indicator.adjusted_profit > 0,
        #     valuation.code.in_(lists)
        # ).order_by(
        #     indicator.roa.desc()  # 按ROA降序排序
        # )
        #

        lists = list(
            get_fundamentals(
                query(valuation.code, indicator.roa).filter(
                    valuation.code.in_(stocks),
                    valuation.pb_ratio > 0,
                    valuation.pb_ratio < 1,
                    indicator.adjusted_profit > 0,
                )
            )
            .sort_values(by="roa", ascending=False)
            .head(50)
            .code
        )
        # lists = list(get_fundamentals(q).head(20).code)  # 获取选股列表
        filter_lowlimit_list = self.utilstool.filter_lowlimit_stock(context, lists)
        final_list = self.utilstool.filter_highlimit_stock(context, filter_lowlimit_list)
        return final_list
