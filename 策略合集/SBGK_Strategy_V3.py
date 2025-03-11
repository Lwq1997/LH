# -*- coding: utf-8 -*-
# 如果你的文件包含中文, 请在文件的第一行使用上面的语句指定你的文件编码

# 用到策略及数据相关API请加入下面的语句(如果要兼容研究使用可以使用 try except导入
from kuanke.user_space_api import *
from Strategy import Strategy
from jqdata import *
from kuanke.wizard import *
from jqlib.technical_analysis import *


class SBGK_Strategy_V3(Strategy):
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
        current_data = get_current_data()
        log.info(self.name, '的选股底池--yes_first_hl_list:', yes_first_hl_list)
        sbgk_stocks = []
        date_now = context.current_dt.strftime("%Y-%m-%d")
        mid_time1 = ' 09:15:00'
        end_times1 = ' 09:26:00'
        start = date_now + mid_time1
        end = date_now + end_times1

        # 首板高开/低开
        for s in yes_first_hl_list:
            all_date = attribute_history(s, 4, '1d', fields=['close', 'volume', 'money'], skip_paused=True)
            # 获取前一日数据
            # 条件一：均价，金额，市值，换手率 收盘获利比例低于7%，成交额小于5.5亿或者大于20亿，或市值小于70亿，大于520亿，过滤
            avg_price_increase_value = all_date['money'][-1] / all_date['volume'][-1] / all_date['close'][-1] * 1.1 - 1
            if avg_price_increase_value < 0.07 or all_date['money'][-1] < 5.5e8 or all_date['money'][-1] > 20e8:
                continue

            # market_cap 总市值(亿元) > 70亿 流通市值(亿元) < 520亿
            turnover_ratio_data = get_valuation(s, start_date=context.previous_date, end_date=context.previous_date,
                                                fields=['turnover_ratio', 'market_cap', 'circulating_market_cap'])
            # 合并条件一剩余的市值等判断，简化空值和范围判断写法
            if turnover_ratio_data.empty or not (70 <= turnover_ratio_data['market_cap'][0] <= 520):
                continue

            # 条件二：左压
            if self.utilstool.rise_low_volume(context, s):
                continue
            # 条件三：高开,开比
            auction_data = get_call_auction(s, start_date=start, end_date=end, fields=['time', 'volume', 'current'])
            if auction_data.empty or auction_data['volume'][0] / all_date['volume'][-1] < 0.03:
                continue
            current_ratio = auction_data['current'][0] / (current_data[s].high_limit / 1.1)
            if current_ratio <= 1 or current_ratio >= 1.06:
                continue

            # 如果股票满足所有条件，则添加到列表中
            sbgk_stocks.append(s)

        log.info('今日首板高开选股：' + ','.join('%s%s' % (s, get_security_info(s).display_name) for s in sbgk_stocks))

        return sbgk_stocks
