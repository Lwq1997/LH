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


class Weak_Cyc_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)
        self.bond_etf = "511260.XSHG"
        # 最小交易额(限制手续费)
        self.min_volume = 2000
        self.pe_mean = 20
        self.time = 0

    def get_stock_sum(self):
        if self.pe_mean < 20:
            self.max_select_count = 4
        elif self.pe_mean < 30:
            self.max_select_count = 2
        else:
            self.max_select_count = 0

    def select(self, context):
        log.info(self.name, '--select函数（弱周期定制）--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        yesterday = context.previous_date
        stocks = get_industry_stocks("HY010", date=yesterday)
        stocks = self.utilstool.filter_basic_stock(context, stocks)
        data = get_fundamentals(
            query(valuation.code, valuation.pe_ratio, valuation.market_cap).filter(valuation.code.in_(stocks)))
        total_market_cap = data.market_cap.sum()
        self.pe_mean = total_market_cap / (1 / data.pe_ratio * data.market_cap).sum()
        self.get_stock_sum()
        stocks = get_fundamentals(
            query(valuation.code, valuation.pe_ratio < 20)
            .filter(
                valuation.code.in_(stocks),
                valuation.market_cap > 200,
                valuation.pe_ratio < 20,
                indicator.roa > 0,
                indicator.gross_profit_margin > 20,  # 毛利
            )
            .order_by(valuation.market_cap.desc())
        ).code
        return list(stocks)

    def adjust(self, context):
        log.info(self.name, '--adject函数（弱周期定制）--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        self.select_list = self.select(context)

        self.select_list = self.utilstool.filter_stocks_by_industry(context, self.select_list, max_industry_stocks=1)
        stocks = self.select_list[: self.max_select_count]

        stocks.append(self.bond_etf)
        rates = [round(1 / (self.max_select_count + 2), 3)] * (len(stocks) - 1)
        rates.append(round(1 - sum(rates), 3))
        subportfolio = context.subportfolios[self.subportfolio_index]

        total_value = subportfolio.total_value
        targets = {stock: total_value * rate for stock, rate in zip(stocks, rates)}

        current_data = get_current_data()
        # 获取当前持仓
        current_positions = subportfolio.long_positions
        log.info(self.name, '的选股列表:', targets, '--当前持仓--', current_positions)

        # 清仓被调出的
        for stock in current_positions:
            if stock not in targets:
                self.utilstool.close_position(context, stock, 0)

        # 先卖出
        for stock, target in targets.items():
            price = current_data[stock].last_price
            value = current_positions[stock].total_amount * price
            log.info(self.name,'--stock--',stock, '--value--', value, '--target--', target, '--price--', price)
            if value - target > self.min_volume and value - target > price * 100:
                self.utilstool.close_position(context, stock, target)

        # self.balance_subportfolios(context)

        # 后买入
        for stock, target in targets.items():
            price = current_data[stock].last_price
            value = current_positions[stock].total_amount * price
            log.info(self.name,'--stock--',stock, '--value--', value, '--target--', target, '--price--', price)
            if target - value > self.min_volume and target - value > price * 100:
                if subportfolio.available_cash > price * 100:
                    self.utilstool.open_position(context, stock, target)
