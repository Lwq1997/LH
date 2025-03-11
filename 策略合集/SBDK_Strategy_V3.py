# -*- coding: utf-8 -*-
# 如果你的文件包含中文, 请在文件的第一行使用上面的语句指定你的文件编码

# 用到策略及数据相关API请加入下面的语句(如果要兼容研究使用可以使用 try except导入
from kuanke.user_space_api import *
from Strategy import Strategy
from jqdata import *
import pandas as pd
from kuanke.wizard import *
from jqlib.technical_analysis import *


class SBDK_Strategy_V3(Strategy):
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

        yes_no_first_hl_list = context.yes_no_first_hl_list
        current_data = get_current_data()
        log.info(self.name, '的选股底池--yes_no_first_hl_list:', yes_no_first_hl_list)
        sbdk_stocks = []
        date_now = context.current_dt.strftime("%Y-%m-%d")
        mid_time1 = ' 09:15:00'
        end_times1 = ' 09:26:00'
        start = date_now + mid_time1
        end = date_now + end_times1

        # 首板低开
        if yes_no_first_hl_list:
            date = self.utilstool.transform_date(context, context.previous_date, 'str')
            # 计算相对位置
            rpd = self.utilstool.get_relative_position_df(context, yes_no_first_hl_list, date, 60)
            rpd = rpd[rpd['rp'] <= 0.5]
            stock_list = list(rpd.index)

            # 低开
            df = get_price(stock_list, end_date=date, frequency='daily', fields=['close'], count=1, panel=False,
                           fill_paused=False, skip_paused=True).set_index('code') if len(
                stock_list) != 0 else pd.DataFrame()
            df['open_pct'] = [current_data[s].day_open / df.loc[s, 'close'] for s in stock_list]
            df = df[(0.955 <= df['open_pct']) & (df['open_pct'] <= 0.97)]  # 低开越多风险越大，选择3个多点即可
            stock_list = list(df.index)

            for s in stock_list:
                prev_day_data = attribute_history(s, 1, '1d', fields=['close', 'volume', 'money'], skip_paused=True)
                if prev_day_data['money'][0] >= 1e8:
                    sbdk_stocks.append(s)

        log.info('今日首板低开选股：' + ','.join('%s%s' % (s, get_security_info(s).display_name) for s in sbdk_stocks))

        return sbdk_stocks
