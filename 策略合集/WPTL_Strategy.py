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
class WPTL_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)
        self.duration_days = params['duration_days']
        self.condition = params['condition']

    def select(self, context):
        log.info(self.name, '--select函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 根据市场温度设置选股条件，选出股票
        self.select_list = self.__get_rank(context)[:self.max_select_count]
        log.error('选股列表:', self.select_list)
        # 编写操作计划
        self.print_trade_plan(context, self.select_list)

    def __get_rank(self, context):
        log.info(self.name, '--get_rank函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        initial_list = super().stockpool(context)
        selected_stocks = {}
        for stock in initial_list:
            log.info("共有股票:", len(initial_list), "当前正在处理", stock)
            # 获取股票过去15天的收盘价、开盘价以及指定时间点（14:50和9:35）价格数据（可按需调整天数）
            price_data_min = attribute_history(stock, count=(self.duration_days) * 243/5+1, fields=['close'],
                                               unit='5m', skip_paused=True, fq='pre')
            # print(price_data_min)

            # 筛选出每天14:50,每天9:35的价格数据
            price_data_min_filter = price_data_min[
                (price_data_min.index.time == datetime.time(9, 35)) |
                (price_data_min.index.time == datetime.time(14, 50))
                # (price_data_min.index.time == datetime.time(9, 31)) |
                # (price_data_min.index.time == datetime.time(15, 00))
                ]

            # 在df1中新增一个日期字段，提取索引中的日期部分，并转换为datetime64[ns]类型
            price_data_min_filter['date'] = pd.to_datetime(price_data_min_filter.index.date)

            # 先提取时间中的小时和分钟部分，组合成新的列用于后续 pivot 操作
            price_data_min_filter['time'] = price_data_min_filter.index.strftime('%H:%M')

            # 使用pivot_table方法进行行转列操作，以日期为索引，将不同时间对应的close值转到不同列
            new_price_data_min_filter = price_data_min_filter.pivot_table(values='close', index='date',
                                                                          columns='time', aggfunc='first')

            # 重命名列，假设新列名分别为time_0931、time_0935、time_1450、time_1500（可根据实际需求调整）
            new_price_data_min_filter = new_price_data_min_filter.rename(columns={
                # '09:31': '0931_close',
                '09:35': '0935_close',
                '14:50': '1450_close'
                # '15:00': '1500_close'
            })

            # new_price_data_min_filter['condition1'] = new_price_data_min_filter['1500_close'].shift(1) < \
            #                                           new_price_data_min_filter['0931_close']
            new_price_data_min_filter['final_condition'] = new_price_data_min_filter['1450_close'].shift(1) < \
                                                      new_price_data_min_filter['0935_close']
            # print(new_price_data_min_filter)

            # if self.condition == 0:
            #     new_price_data_min_filter['final_condition'] = new_price_data_min_filter['condition1']
            # elif self.condition == 1:
            #     new_price_data_min_filter['final_condition'] = new_price_data_min_filter['condition2']
            # else:
            #     # # 两个条件同时满足才为True
            #     new_price_data_min_filter['final_condition'] = new_price_data_min_filter['condition1'] & new_price_data_min_filter['condition2']

            # 将condition2列转换为整数类型（True为1，False为0），便于后续计算
            new_price_data_min_filter['final_condition_int'] = new_price_data_min_filter['final_condition'].astype(
                int)

            # 计算连续满足条件的天数
            new_price_data_min_filter['continuous_days'] = new_price_data_min_filter['final_condition_int'].groupby(
                (
                        new_price_data_min_filter['final_condition_int'].shift(1) != new_price_data_min_filter[
                    'final_condition_int'])
                .cumsum()
            ).cumsum()
            # pd.set_option('display.float_format', lambda x: '%.2f' % x)
            # # 显示所有列
            # pd.set_option('display.max_columns', None)
            # pd.set_option('display.max_rows', None)
            # log.error(new_price_data_min_filter)

            if (new_price_data_min_filter['continuous_days'] >= self.duration_days).any():  # 检查是否有连续7天以上满足条件的情况
                selected_stocks[stock] = new_price_data_min_filter['continuous_days'].max()

        log.error(selected_stocks)
        sorted_items = sorted(selected_stocks.items(), key=lambda x: x[1])  # 按照值进行排序，返回包含(key, value)元组的列表
        result_stock = [item[0] for item in sorted_items[:self.max_select_count]]  # 取前N个元组中的key

        return result_stock
