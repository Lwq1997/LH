# -*- coding: utf-8 -*-
# 如果你的文件包含中文, 请在文件的第一行使用上面的语句指定你的文件编码

# 用到策略及数据相关API请加入下面的语句(如果要兼容研究使用可以使用 try except导入
import datetime
import math
import numpy as np
import pandas as pd
import pickle
import requests
import talib

from kuanke.user_space_api import *
from kuanke.wizard import *
from jqdata import *
from jqfactor import *
from jqlib.technical_analysis import *
from scipy.linalg import inv
from prettytable import PrettyTable
from WPETF_Strategy import WPETF_Strategy
from XSZ_GJT_Strategy import XSZ_GJT_Strategy
from Strategy import Strategy

import warnings
import datetime as dt


# DSZMX策略
class DXW_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)
        self.new_days = 375  # 已上市天数
        self.no_trading_today_signal = False
        self.market_temperature = "warm"
        self.highest = 50
        self.small_stock_count = 9
        self.big_stock_count = 6
        self.singal_str = 'Unknown'
        self.ETF_pool = []
        self.foreign_ETF = [
            '518880.XSHG',
            '513030.XSHG',
            '513100.XSHG',
            '164824.XSHE',
            '159866.XSHE',
        ]

    def singal(self, context):
        dt_last = context.previous_date
        log.info(self.name, '--singal函数开始运行--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        B_stocks = self.stockpool(context, 1, "000300.XSHG", is_kcbj=True, is_st=True, is_paused=False,
                                  is_lowlimit=False, is_highlimit=False)
        # 过滤次新股
        B_stocks = self.utilstool.filter_new_stock(context, B_stocks, self.new_days)
        S_stocks = self.stockpool(context, 1, '399101.XSHE', is_kcbj=True, is_st=True, is_paused=False,
                                  is_lowlimit=False, is_highlimit=False)

        # 过滤次新股
        S_stocks = self.utilstool.filter_new_stock(context, S_stocks, self.new_days)

        q = query(
            valuation.code, valuation.circulating_market_cap
        ).filter(
            valuation.code.in_(B_stocks)
        ).order_by(
            valuation.circulating_market_cap.desc()
        )
        df = get_fundamentals(q)
        Blst = list(df.code)[:20]

        q = query(
            valuation.code, valuation.circulating_market_cap
        ).filter(
            valuation.code.in_(S_stocks)
        ).order_by(
            valuation.circulating_market_cap.asc()
        )
        df = get_fundamentals(q, date=dt_last)
        Slst = list(df.code)[:20]

        B_ratio = get_price(Blst, end_date=dt_last, frequency='1d', fields=['close'], count=10, panel=False
                            ).pivot(index='time', columns='code', values='close')
        change_BIG = (B_ratio.iloc[-1] / B_ratio.iloc[0] - 1) * 100
        A1 = np.array(change_BIG)
        A1 = np.nan_to_num(A1)
        B_mean = np.mean(A1)

        S_ratio = get_price(Slst, end_date=dt_last, frequency='1d', fields=['close'], count=10, panel=False
                            ).pivot(index='time', columns='code', values='close')
        change_SMALL = (S_ratio.iloc[-1] / S_ratio.iloc[0] - 1) * 100
        A1 = np.array(change_SMALL)
        A1 = np.nan_to_num(A1)
        S_mean = np.mean(A1)

        if B_mean > S_mean and B_mean > 0:
            if B_mean > 5:
                self.singal_str = 'small'
            else:
                self.singal_str = 'big'
        elif B_mean < S_mean and S_mean > 0:
            self.singal_str = 'small'
        else:
            self.singal_str = 'etf'
            deltaday = 20
            self.ETF_pool = self.fun_delNewShare(context, self.foreign_ETF, deltaday)
            if len(self.ETF_pool) == 0:
                self.ETF_pool = self.fun_delNewShare(context, ['511010.XSHG'], deltaday)

        log.info(self.name, '--singal函数运行完成，运行后的结果：', self.singal_str, '--now--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # if B_mean > 10 or S_mean > 10:
        #     print('无敌好行情')
        #     if B_mean > S_mean:
        #         print('开大')
        #         choice = B_stocks
        #         target_list1 = ROIC_BIG(context, choice)
        #         target_list2 = BIG(context, choice)
        #         target_list3 = BM(context, choice)
        #         target_list = target_list3 + target_list1 + target_list2
        #         target_list = list(set(target_list))
        #     else:
        #         print('开小')
        #         choice = S_stocks
        #         target_list = SMALL(context, choice)[:self.stock_num * 3]
        # elif B_mean > S_mean and B_mean > 0:
        #     print('开大')
        #     choice = B_stocks
        #     target_list2 = ROIC_BIG(context, choice)
        #     target_list1 = BIG(context, choice)
        #     target_list3 = BM(context, choice)
        #     target_list = target_list1 + target_list2 + target_list3
        #     target_list = list(set(target_list))
        #
        # elif B_mean < S_mean and S_mean > 0:
        #     print('开小')
        #     choice = S_stocks
        #     target_list = SMALL(context, choice)[:self.stock_num * 3]
        # else:
        #     print('开外盘')
        #     target_list = self.foreign_ETF

    def select(self, context):
        self.select_list = self.__get_rank(context)[:self.max_select_count]
        log.error('选股列表:', self.select_list)
        self.print_trade_plan(context, self.select_list)

    def __get_rank(self, context):

        log.info(self.name, '--get_rank函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        target_list = []

        B_stocks = self.stockpool(context, 1, '000300.XSHG')
        # 过滤次新股
        B_stocks = self.utilstool.filter_new_stock(context, B_stocks, self.new_days)

        S_stocks = self.stockpool(context, 1, '399101.XSHE')
        # 过滤次新股
        S_stocks = self.utilstool.filter_new_stock(context, S_stocks, self.new_days)

        if self.singal_str == 'big':
            target_list = self.White_Horse(context, B_stocks)
        elif self.singal_str == 'small':
            target_list = self.SMALL(context, S_stocks)
        elif self.singal_str == 'etf':
            # target_list = self.foreign_ETF
            target_list = self.ETF_pool
        return target_list

    ## 开盘前运行函数
    def White_Horse(self, context, B_stocks):

        self.market_temperature = self.utilstool.Market_temperature(context, self.market_temperature)

        log.info(self.name, '--White_Horse函数开始运行，当前时长温度:', self.market_temperature, '--now--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        all_stocks = B_stocks
        if self.market_temperature == "cold":
            q = query(
                valuation.code,
            ).filter(
                # TODO
                # valuation.pb_ratio > 0,
                # valuation.pb_ratio < 1,
                # cash_flow.subtotal_operate_cash_inflow > 0,  # 经营活动现金流入小计
                # indicator.adjusted_profit > 2.5e8,  # 扣除非经常损益后的净利润(元)                          #>=2.5亿
                # income.operating_revenue > 10e8,  # 营业收入(元)                                          #>=10亿
                # income.net_profit > 2.5e8,  # 净利润(元)                                            #>=2.5亿
                # cash_flow.subtotal_operate_cash_inflow / indicator.adjusted_profit > 2.0,  # 经营活动现金流入小计/扣除非经常损益后的净利润(元)
                # indicator.inc_return > 1.5,  # 净资产收益率(扣除非经常损益)(%)
                # indicator.inc_net_profit_year_on_year > -15,  # 净利润同比增长率(%)
                # valuation.code.in_(initial_list)
                valuation.pb_ratio > 0,
                valuation.pb_ratio < 1,
                cash_flow.subtotal_operate_cash_inflow > 0,
                indicator.adjusted_profit > 0,
                cash_flow.subtotal_operate_cash_inflow / indicator.adjusted_profit > 2.0,
                indicator.inc_return > 1.5,
                indicator.inc_net_profit_year_on_year > -15,
                valuation.code.in_(all_stocks)
            ).order_by(
                (indicator.roa / valuation.pb_ratio).desc()
            ).limit(
                self.big_stock_count
            )
        elif self.market_temperature == "warm":
            q = query(
                valuation.code,
            ).filter(
                # valuation.pb_ratio > 0,
                # valuation.pb_ratio < 1,
                # cash_flow.subtotal_operate_cash_inflow > 0,
                # indicator.adjusted_profit > 2.5e8,  # 扣除非经常损益后的净利润(元)                          #>=2.5亿
                # income.operating_revenue > 10e8,  # 营业收入(元)                                          #>=10亿
                # income.net_profit > 2.5e8,  # 净利润(元)                                            #>=2.5亿
                # cash_flow.subtotal_operate_cash_inflow / indicator.adjusted_profit > 1.0,
                # indicator.inc_return > 2.0,
                # indicator.inc_net_profit_year_on_year > 0,
                # valuation.code.in_(initial_list)

                valuation.pb_ratio > 0,
                valuation.pb_ratio < 1,
                cash_flow.subtotal_operate_cash_inflow > 0,
                indicator.adjusted_profit > 0,
                cash_flow.subtotal_operate_cash_inflow / indicator.adjusted_profit > 1.0,
                indicator.inc_return > 2.0,
                indicator.inc_net_profit_year_on_year > 0,
                valuation.code.in_(all_stocks)
            ).order_by(
                (indicator.roa / valuation.pb_ratio).desc()
            ).limit(
                self.big_stock_count
            )
        elif self.market_temperature == "hot":
            q = query(
                valuation.code,
            ).filter(
                # valuation.pb_ratio > 3,
                # cash_flow.subtotal_operate_cash_inflow > 0,
                # indicator.adjusted_profit > 2.5e8,  # 扣除非经常损益后的净利润(元)                          #>=2.5亿
                # income.operating_revenue > 10e8,  # 营业收入(元)                                          #>=10亿
                # income.net_profit > 2.5e8,  # 净利润(元)                                            #>=2.5亿
                # cash_flow.subtotal_operate_cash_inflow / indicator.adjusted_profit > 0.5,
                # indicator.inc_return > 3.0,
                # indicator.inc_net_profit_year_on_year > 20,
                # valuation.code.in_(initial_list)
                valuation.pb_ratio > 3,
                cash_flow.subtotal_operate_cash_inflow > 0,
                indicator.adjusted_profit > 0,
                cash_flow.subtotal_operate_cash_inflow / indicator.adjusted_profit > 0.5,
                indicator.inc_return > 3.0,
                indicator.inc_net_profit_year_on_year > 20,
                valuation.code.in_(all_stocks)
            ).order_by(
                indicator.roa.desc()
            ).limit(
                self.big_stock_count
            )

        check_out_lists = list(get_fundamentals(q).code)
        return check_out_lists

    def SMALL(self, context, S_stocks):
        all_stocks1 = S_stocks
        all_stocks2 = S_stocks
        log.info(self.name, '--SMALL函数开始运行--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        q = query(valuation.code, valuation.market_cap).filter(valuation.code.in_(all_stocks1),
                                                               valuation.market_cap.between(5, 30)).order_by(
            valuation.market_cap.asc())
        df_fun = get_fundamentals(q)[:50]
        final_list_1 = list(df_fun.code)

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
            valuation.code.in_(all_stocks2),
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

        target_list = list(dict.fromkeys(final_list_1 + final_list_2))
        target_list = target_list[:self.small_stock_count]
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

    ####
    def fun_delNewShare(self, context, equity, deltaday):
        log.info(self.name, '--fun_delNewShare函数开始运行--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        deltaDate = context.current_dt.date() - dt.timedelta(deltaday)
        tmpList = []
        for stock in equity:
            if get_security_info(stock).start_date < deltaDate:
                tmpList.append(stock)
        return tmpList
