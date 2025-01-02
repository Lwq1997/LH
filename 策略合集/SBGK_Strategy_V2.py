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
class SBGK_Strategy_V2(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)
        self.n_days_limit_up_list = []

    def select(self, context):
        log.info(self.name, '--select函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 根据市场温度设置选股条件，选出股票
        self.select_list = self.__get_rank(context)
        # 编写操作计划
        self.print_trade_plan(context, self.select_list)

    def __get_rank(self, context):
        log.info(self.name, '--get_rank函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        yes_first_hl_list = context.yes_first_hl_list
        log.info(self.name, '--yes_first_hl_list:', yes_first_hl_list)
        sbgk_stocks = []
        date_now = context.current_dt.strftime("%Y-%m-%d")
        mid_time1 = ' 09:15:00'
        end_times1 = ' 09:26:00'
        start = date_now + mid_time1
        end = date_now + end_times1

        # 首板高开/低开
        for s in yes_first_hl_list:
            zyts = self.utilstool.calculate_zyts(context, s)

            his_day = zyts if zyts > 4 else 4
            all_date = attribute_history(s, his_day, '1d', fields=['close', 'volume', 'money'], skip_paused=True)
            # 获取前一日数据
            # prev_day_data = attribute_history(s, 1, '1d', fields=['close', 'volume', 'money'], skip_paused=True)
            # 条件一：均价，金额，市值，换手率 收盘获利比例低于7%，成交额小于5.5亿或者大于20亿，或市值小于70亿，大于520亿，过滤
            avg_price_increase_value = all_date['money'][-1] / all_date['volume'][-1] / all_date['close'][
                -1] * 1.1 - 1
            if avg_price_increase_value < 0.07 or all_date['money'][-1] < 5e8 or all_date['money'][-1] > 20e8:
                continue

            turnover_ratio_data = get_valuation(s, start_date=context.previous_date, end_date=context.previous_date,
                                                fields=['turnover_ratio', 'market_cap', 'circulating_market_cap'])
            # 合并条件一剩余的市值等判断，简化空值和范围判断写法
            if turnover_ratio_data.empty or not (70 <= turnover_ratio_data['market_cap'][0] <= 520):
                continue

            # 条件二：左压
            # 简化成交量同步放大判断
            # if s == '002046.XSHE':
            #     log.debug('被条件二左压过滤：zyts:', zyts, '--交易量:', all_date['volume'][-1], '--最大交易量:',
            #               max(all_date['volume'][:-1]),'--明细:',all_date)

            if zyts < 2 or all_date['volume'][-1] <= max(all_date['volume'][:-1]) * 0.9:
                continue

            # 条件三：高开,开比
            auction_data = get_call_auction(s, start_date=start, end_date=end, fields=['time', 'volume', 'current'])
            if auction_data.empty or auction_data['volume'][0] / all_date['volume'][-1] < 0.03:
                continue
            current_ratio = auction_data['current'][0] / all_date['close'][-1]
            if not (1 < current_ratio < 1.06):
                continue

            # 条件四：过滤前面三天涨幅超过20%的票
            if len(all_date) < 4 or (all_date['close'][-1] - all_date['close'][-4]) / all_date['close'][0] > 0.20:
                continue

            # 如果股票满足所有条件，则添加到列表中
            sbgk_stocks.append(s)

        log.info('今日首板高开选股：' + ','.join('%s%s'%(s, get_security_info(s).display_name) for s in sbgk_stocks))

        return sbgk_stocks
