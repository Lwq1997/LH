# -*- coding: utf-8 -*-
# 如果你的文件包含中文, 请在文件的第一行使用上面的语句指定你的文件编码

# 用到策略及数据相关API请加入下面的语句(如果要兼容研究使用可以使用 try except导入
from kuanke.user_space_api import *
from Strategy import Strategy
from jqdata import *
from kuanke.wizard import *
from jqlib.technical_analysis import *


class OGT_Strategy(Strategy):
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

        two_hl_list = context.two_hl_list
        three_hl_list = context.three_hl_list
        current_data = get_current_data()
        log.info(self.name, '的选股底池--two_hl_list:', two_hl_list)
        log.info(self.name, '的选股底池--three_hl_list:', three_hl_list)
        ogt_stocks = []
        date_now = context.current_dt.strftime("%Y-%m-%d")
        mid_time1 = ' 09:15:00'
        end_times1 = ' 09:26:00'
        start = date_now + mid_time1
        end = date_now + end_times1

        # 首板高开/低开
        for s in two_hl_list:
            # 条件一：均价，金额，市值，换手率
            prev_day_data = attribute_history(s, 1, '1d', fields=['close', 'volume', 'money'], skip_paused=True)
            avg_price_increase_value = prev_day_data['money'][0] / prev_day_data['volume'][0] / prev_day_data['close'][
                0] * 1.1 - 1
            # 如果平均价格涨幅小于0.07或者前一个交易日的成交金额小于7亿或者大于20亿，则跳过
            if avg_price_increase_value < 0.07 or prev_day_data['money'][0] < 7e8 or prev_day_data['money'][0] > 30e8:
                continue

            # 条件二: market_cap 总市值(亿元) > 70亿, 流通市值(亿元) < 300亿
            turnover_ratio_data = get_valuation(s, start_date=context.previous_date, end_date=context.previous_date,
                                                fields=['turnover_ratio', 'market_cap', 'circulating_market_cap'])
            if turnover_ratio_data.empty or not (70 <= turnover_ratio_data['market_cap'][0] <= 520) or \
                    turnover_ratio_data['circulating_market_cap'][0] > 300:
                continue

            yesterday_turnover_ratio = turnover_ratio_data['turnover_ratio'][0]
            if yesterday_turnover_ratio < 10 or yesterday_turnover_ratio > 30:
                continue

            # 条件三：昨日涨停的成交量为近100日的最大成交量
            # 获取昨日成交量
            yesterday_volume = prev_day_data['volume'][0]

            log.info(f'{s}昨日成交量{yesterday_volume}')
            # 获取过去100个交易日的成交量
            past_volume_data = attribute_history(s, 100, '1d', fields=['volume'], skip_paused=True)
            if past_volume_data.empty:
                continue
            max_past_volume = past_volume_data['volume'].max()
            log.info(f'{s}最大成交量{max_past_volume}')
            if yesterday_volume < max_past_volume:
                continue

            # 条件四： 昨日收盘时封单金额需大于流通市值的2%
            # 获取昨日收盘时的封单金额
            # 使用 get_ticks 获取昨日最后一笔的盘口数据

            edate = context.previous_date
            end_time = str(edate) + ' ' + '15:00:00'
            ticks = get_ticks(s, end_dt=end_time, count=1, fields=['time', 'a1_v', 'a1_p', 'b1_v', 'b1_p'], skip=False,
                              df=True)
            if len(ticks) == 0:
                continue

            bid_volume = ticks['b1_p'].iloc[0]
            bid_price = ticks['b1_v'].iloc[0]
            # 计算封单金额
            order_amount = bid_volume * bid_price
            # 获取流通市值
            circulating_market_cap = turnover_ratio_data['circulating_market_cap'][0]
            # 计算封单金额占流通市值的比例
            order_ratio = order_amount / (circulating_market_cap * 100000000)
            if order_ratio < 0.01:
                continue

            df = get_price(s, end_date=context.previous_date, frequency='daily', fields=['low', 'close', 'low_limit'],
                           count=10,
                           panel=False,
                           fill_paused=False, skip_paused=False)
            low_limit_count = len(df[df.close == df.low_limit])
            if low_limit_count >= 1:
                continue

            # 将符合条件的股票添加到保存的股票列表中
            ogt_stocks.append(s)

        for s in three_hl_list:
            # 过滤前面三天涨幅超过18%的票
            price_data = attribute_history(s, 4, '1d', fields=['close'], skip_paused=True)
            increase_ratio = (price_data['close'][-1] - price_data['close'][0]) / price_data['close'][0]
            if len(price_data) < 4 or increase_ratio > 0.18:
                continue

            # 条件一：均价，金额，市值，换手率
            prev_day_data = attribute_history(s, 1, '1d', fields=['close', 'volume', 'money'], skip_paused=True)
            avg_price_increase_value = prev_day_data['money'][0] / prev_day_data['volume'][0] / prev_day_data['close'][
                0] * 1.1 - 1
            # 如果平均价格涨幅小于0.07或者前一个交易日的成交金额小于7亿或者大于20亿，则跳过
            if avg_price_increase_value < 0.07 or prev_day_data['money'][0] < 7e8 or prev_day_data['money'][0] > 20e8:
                continue
            # 如果换手率为空或者市值小于70，则跳过
            turnover_ratio_data = get_valuation(s, start_date=context.previous_date, end_date=context.previous_date,
                                                fields=['turnover_ratio', 'market_cap', 'circulating_market_cap'])
            if turnover_ratio_data.empty or turnover_ratio_data['market_cap'][0] < 70 or \
                    turnover_ratio_data['circulating_market_cap'][0] > 300:
                continue
            # 如果近期有跌停，则跳过
            df = get_price(s, end_date=context.previous_date, frequency='daily', fields=['low', 'close', 'low_limit'],
                           count=10,
                           panel=False,
                           fill_paused=False, skip_paused=False)
            low_limit_count = len(df[df.close == df.low_limit])
            if low_limit_count >= 1:
                continue

            # 条件二：左压
            zyts = self.utilstool.calculate_zyts(context, s)
            volume_data = attribute_history(s, zyts, '1d', fields=['volume'], skip_paused=True)
            if len(volume_data) < 2 or volume_data['volume'][-1] <= max(volume_data['volume'][:-1]) * 0.90:
                continue

            # 将符合条件的股票添加到保存的股票列表中
            ogt_stocks.append(s)

        ogt_stocks = list(set(ogt_stocks))
        log.info('今日一进二选股：' + ','.join('%s%s' % (s, get_security_info(s).display_name) for s in ogt_stocks))

        return ogt_stocks
