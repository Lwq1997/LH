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
class SBGK_Strategy(Strategy):
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
        log.info('yes_first_hl_list:',yes_first_hl_list)
        log.info('yes_first_no_hl_list:', yes_first_no_hl_list)
        sbdk_stocks = []
        sbgk_stocks = []
        rzq_stocks = []
        current_data = get_current_data()
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
            if rp <= 0.5 and money >= 1e8:
                auction_data = get_call_auction(s, start_date=start, end_date=end, fields=['time', 'current'])
                if not auction_data.empty:
                    current_ratio = auction_data['current'][0] / close
                    if current_ratio <= 0.97 and current_ratio >= 0.955:
                        sbdk_stocks.append(s)
                        continue

            # 条件一：均价，金额，市值，换手率 收盘获利比例低于7%，成交额小于5.5亿或者大于20亿，或市值小于70亿，大于520亿，过滤
            prev_day_data = attribute_history(s, 1, '1d', fields=['close', 'volume', 'money'], skip_paused=True)
            avg_price_increase_value = prev_day_data['money'][0] / prev_day_data['volume'][0] / prev_day_data['close'][
                0] * 1.1 - 1
            if avg_price_increase_value < 0.07 or prev_day_data['money'][0] < 5.5e8 or prev_day_data['money'][0] > 20e8:
                continue
            turnover_ratio_data = get_valuation(s, start_date=context.previous_date, end_date=context.previous_date,
                                                fields=['turnover_ratio', 'market_cap', 'circulating_market_cap'])
            if turnover_ratio_data.empty or turnover_ratio_data['market_cap'][0] < 70 or \
                    turnover_ratio_data['circulating_market_cap'][0] > 520:
                continue

            # 条件二：高开,开比
            auction_data = get_call_auction(s, start_date=start, end_date=end, fields=['time', 'volume', 'current'])
            # print([s,auction_data['volume'][0],prev_day_data['volume'][-1]])
            if auction_data.empty or auction_data['volume'][0] / prev_day_data['volume'][-1] < 0.03:
                continue
            current_ratio = auction_data['current'][0] / prev_day_data['close'][-1]
            if current_ratio <= 1 or current_ratio >= 1.06:
                continue

            # 条件三：左压
            hst = attribute_history(s, 101, '1d', fields=['high', 'volume'], skip_paused=True)  # 获取历史数据
            prev_high = hst['high'].iloc[-1]  # 计算前一天的高点
            zyts_0 = next((i - 1 for i, high in enumerate(hst['high'][-3::-1], 2) if high >= prev_high),
                          100)  # 计算zyts_0
            zyts = zyts_0 + 5
            volume_data = hst['volume'][-zyts:]  # 获取高点以来的成交量数据
            # 检查今天的成交量是否同步放大
            if len(volume_data) < 2 or volume_data[-1] <= max(volume_data[:-1]) * 0.9:
                continue

            # 如果股票满足所有条件，则添加到列表中
            sbgk_stocks.append(s)

        # 弱转强
        for s in yes_first_no_hl_list:
            # 过滤前面三天涨幅超过28%的票
            prev_day_data = attribute_history(s, 4, '1d', fields=['open', 'close', 'volume', 'money'], skip_paused=True)
            increase_ratio = (prev_day_data['close'][-1] - prev_day_data['close'][0]) / prev_day_data['close'][0]
            if increase_ratio > 0.28:
                continue

            # 过滤前一日收盘价小于开盘价5%以上的票
            open_close_ratio = (prev_day_data['close'][-1] - prev_day_data['open'][-1]) / prev_day_data['open'][-1]
            if open_close_ratio < -0.05:
                continue

            # 条件一：均价，金额，市值，换手率 收盘获利比例低于4%，成交额小于3亿或者大于19亿，或市值小于70亿，大于520亿，过滤
            avg_price_increase_value = prev_day_data['money'][-1] / prev_day_data['volume'][-1] / \
                                       prev_day_data['close'][
                                           -1] - 1
            if avg_price_increase_value < -0.04 or prev_day_data['money'][-1] < 3e8 or prev_day_data['money'][
                -1] > 19e8:
                continue
            turnover_ratio_data = get_valuation(s, start_date=context.previous_date, end_date=context.previous_date,
                                                fields=['turnover_ratio', 'market_cap', 'circulating_market_cap'])
            if turnover_ratio_data.empty or turnover_ratio_data['market_cap'][0] < 70 or \
                    turnover_ratio_data['circulating_market_cap'][0] > 520:
                continue

            # 条件二：高开,开比
            auction_data = get_call_auction(s, start_date=start, end_date=end, fields=['time', 'volume', 'current'])
            if auction_data.empty or auction_data['volume'][0] / prev_day_data['volume'][-1] < 0.03:
                continue
            current_ratio = auction_data['current'][0] / (
                        current_data[s].high_limit / 1.1)  # prev_day_data['close'][-1]
            if current_ratio <= 0.98 or current_ratio >= 1.09:
                continue

            # 条件三：左压
            hst = attribute_history(s, 101, '1d', fields=['high', 'volume'], skip_paused=True)  # 获取历史数据
            prev_high = hst['high'].iloc[-1]  # 计算前一天的高点
            zyts_0 = next((i - 1 for i, high in enumerate(hst['high'][-3::-1], 2) if high >= prev_high),
                          100)  # 计算zyts_0
            zyts = zyts_0 + 5
            volume_data = hst['volume'][-zyts:]  # 获取高点以来的成交量数据
            # 检查今天的成交量是否同步放大
            if len(volume_data) < 2 or volume_data[-1] <= max(volume_data[:-1]) * 0.9:
                continue

            rzq_stocks.append(s)

        qualified_stocks = sbgk_stocks + sbdk_stocks + rzq_stocks
        if qualified_stocks:
            log.info('今日选股：' + str(qualified_stocks))
            log.info('首板高开：' + str(sbgk_stocks))
            log.info('首板低开：' + str(sbdk_stocks))
            log.info('弱转强：' + str(rzq_stocks))

        return qualified_stocks

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