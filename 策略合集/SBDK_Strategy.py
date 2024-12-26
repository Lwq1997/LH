# -*- coding: utf-8 -*-
# 如果你的文件包含中文, 请在文件的第一行使用上面的语句指定你的文件编码

# 用到策略及数据相关API请加入下面的语句(如果要兼容研究使用可以使用 try except导入
from kuanke.user_space_api import *
from Strategy import Strategy
from jqdata import *
import datetime
from kuanke.wizard import *
import pandas as pd
from jqlib.technical_analysis import *


# 白马股攻防转换策略（BMZH策略）
class SBDK_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)
        self.n_days_limit_up_list = []

    def select(self, context):
        log.info(self.name, '--select函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 根据市场温度设置选股条件，选出股票
        self.select_list = self.__get_rank(context)
        # 编写操作计划
        # self.print_trade_plan(context, self.select_list)

    def __get_rank(self, context):
        log.info(self.name, '--get_rank函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        yes_first_hl_list, yes_first_no_hl_list = self.prepare_stock_list(context)
        log.info(self.name, '--yes_first_hl_list:', yes_first_hl_list)
        sbdk_stocks = []
        date_now = context.current_dt.strftime("%Y-%m-%d")
        mid_time1 = ' 09:15:00'
        end_times1 = ' 09:26:00'
        start = date_now + mid_time1
        end = date_now + end_times1

        # 首板高开/低开
        for s in yes_first_hl_list:
            # 首版低开条件 股票处于一段时间内相对位置<50% 低开3%-4%
            history_data = attribute_history(s, 60, '1d', fields=['close', 'high', 'low', 'money'], skip_paused=True)
            close = history_data['close'][-1]
            high = history_data['high'].max()
            low = history_data['low'].min()
            rp = (close - low) / (high - low)
            money = history_data['money'][-1]
            log.debug('stock:', s, '--rp:', rp, '--money:', money)
            if rp <= 0.5 and money >= 1e8:
                auction_data = get_call_auction(s, start_date=start, end_date=end, fields=['time', 'current'])
                if not auction_data.empty:
                    current_ratio = auction_data['current'][0] / close
                    log.debug('stock:', s, '--auction_data:', auction_data, '--current_ratio:', current_ratio)
                    if current_ratio <= 0.97 and current_ratio >= 0.955:
                        sbdk_stocks.append(s)
                        continue

        log.info('今日首板低开选股：' + str(sbdk_stocks))

        return sbdk_stocks

    def prepare_stock_list(self, context):
        initial_list = super().stockpool(context)
        yesterday = context.previous_date

        # 首次运行，添加前2天的数据
        if not self.n_days_limit_up_list:
            # 获取最近一次交易日
            latest_trade_dates = get_trade_days(end_date=yesterday, count=3)[:-1]
            for day in latest_trade_dates:
                self.n_days_limit_up_list.append(self.utilstool.get_hl_stock(context, initial_list, day))

        # 昨日涨停
        yes_hl_list = self.utilstool.get_hl_stock(context, initial_list, yesterday)
        self.n_days_limit_up_list.append(yes_hl_list)

        # print(self.n_days_limit_up_list)
        # 前1日曾涨停
        hl1_list = set(self.n_days_limit_up_list[-2])
        # 昨日首板
        yes_first_hl_list = [stock for stock in yes_hl_list if stock not in hl1_list]

        # 昨日曾涨停但未封板
        hl_list2 = self.utilstool.get_ever_hl_stock2(context, initial_list, yesterday)
        # 前一日没涨停，昨日曾涨停但未封板
        yes_first_no_hl_list = [stock for stock in hl_list2 if stock not in hl1_list]

        # 移除无用的数据
        self.n_days_limit_up_list.pop(0)

        return yes_first_hl_list, yes_first_no_hl_list
