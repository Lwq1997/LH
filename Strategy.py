# -*- coding: utf-8 -*-
# 如果你的文件包含中文, 请在文件的第一行使用上面的语句指定你的文件编码

# 用到策略及数据相关API请加入下面的语句(如果要兼容研究使用可以使用 try except导入
from kuanke.user_space_api import *
from jqdata import *
from jqfactor import get_factor_values
import datetime
from kuanke.wizard import *
import numpy as np
import pandas as pd
import talib
from datetime import date as dt
import math
import talib as tl
from jqlib.technical_analysis import *
from scipy.linalg import inv
import pickle
import requests
import datetime as datet
from prettytable import PrettyTable
import inspect
from UtilsToolClass import UtilsToolClass


# 策略基类
class Strategy:
    def __init__(self, context, subportfolio_index, name, params):
        self.subportfolio_index = subportfolio_index
        self.name = name
        self.params = params

        self.trade_num = 0
        self.win_num = 0
        self.win_lose_rate = 0
        self.sharp = 0
        self.portfolio_value = pd.DataFrame(columns=['date', 'total_value'])
        self.strategyID = self.params['strategyID'] if 'strategyID' in self.params else ''
        self.inout_cash = 0

        self.max_hold_count = self.params['max_hold_count'] if 'max_hold_count' in self.params else 1  # 最大持股数
        self.max_select_count = self.params['max_select_count'] if 'max_select_count' in self.params else 5  # 最大输出选股数
        self.hold_limit_days = self.params['hold_limit_days'] if 'hold_limit_days' in self.params else 20  # 计算最近持有列表的天数
        self.use_empty_month = self.params['use_empty_month'] if 'use_empty_month' in self.params else False  # 是否有空仓期
        self.empty_month = self.params['empty_month'] if 'empty_month' in self.params else []  # 空仓月份
        self.use_stoplost = self.params['use_stoplost'] if 'use_stoplost' in self.params else False  # 是否使用止损
        self.stoplost_silent_days = self.params[
            'stoplost_silent_days'] if 'stoplost_silent_days' in self.params else 20  # 止损后不交易的天数
        self.stoplost_level = self.params['stoplost_level'] if 'stoplost_level' in self.params else 0.2  # 止损的下跌幅度（按买入价）

        self.select_list = []
        self.hold_list = []  # 昨收持仓
        self.history_hold_list = []  # 最近持有列表
        self.not_buy_again_list = []  # 最近持有不再购买列表
        self.yestoday_high_limit_list = []  # 昨日涨停列表
        self.stoplost_date = None  # 止损日期，为None是表示未进入止损

        self.utilstool = UtilsToolClass()
        self.utilstool.set_params(name, subportfolio_index)

        self.bought_stocks = {}  # 记录补跌的股票和金额
        self.is_stoplost_or_highlimit = False  # 记录是否卖出过止损的股票

        # 行业列表
        # self.industry_list = []
        # 概念列表
        # self.concept_list = []

        # 设置关仓变量，1/4月不交易
        self.no_trading_today_signal = self.params[
            'no_trading_today_signal'] if 'no_trading_today_signal' in self.params else False

    # 每天准备工作
    def day_prepare(self, context):
        log.info(self.name, '--day_prepare选股前的准备工作函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        subportfolio = context.subportfolios[self.subportfolio_index]

        # 获取昨日持股列表
        self.hold_list = list(subportfolio.long_positions)

        # # 获取最近一段时间持有过的股票列表，放入一个新的列表中
        # self.history_hold_list.append(self.hold_list)
        # # 这个列表只维护最近hold_limit_days天的股票池
        # if len(self.history_hold_list) >= self.hold_limit_days:
        #     self.history_hold_list = self.history_hold_list[-self.hold_limit_days:]
        # temp_set = set()
        # for lists in self.history_hold_list:
        #     for stock in lists:
        #         temp_set.add(stock)
        # # 用于记录最近一段时间内曾经持有的股票，避免重复买入。
        # self.not_buy_again_list = list(temp_set)

        # 获取昨日持股中的涨停列表
        if self.hold_list != []:
            df = get_price(self.hold_list, end_date=context.previous_date, frequency='daily',
                           fields=['close', 'high_limit'], count=1, panel=False, fill_paused=False)
            df = df[df['close'] == df['high_limit']]
            self.yestoday_high_limit_list = list(df.code)
        else:
            self.yestoday_high_limit_list = []

        # 检查空仓期
        self.check_empty_month(context)
        # 检查止损
        self.check_stoplost(context)

    # 基础股票池-全市场选股
    def stockpool(self, context, pool_id=1, index=None, is_filter_kcbj=True, is_filter_st=True, is_filter_paused=True,
                  is_filter_highlimit=True,
                  is_filter_lowlimit=True, is_filter_new=True):
        log.info(self.name, '--stockpool函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        if index is None:
            lists = list(get_all_securities(types=['stock'], date=context.previous_date).index)
        else:
            lists = list(get_index_stocks(index))

        if pool_id == 0:
            pass
        elif pool_id == 1:
            if is_filter_kcbj:
                lists = self.utilstool.filter_kcbj_stock(context, lists)
            if is_filter_st:
                lists = self.utilstool.filter_st_stock(context, lists)
            if is_filter_paused:
                lists = self.utilstool.filter_paused_stock(context, lists)
            if is_filter_highlimit:
                lists = self.utilstool.filter_highlimit_stock(context, lists)
            if is_filter_lowlimit:
                lists = self.utilstool.filter_lowlimit_stock(context, lists)
            if is_filter_new:
                lists = self.utilstool.filter_new_stock(context, lists, days=375)

        return lists

    # 按指数选股票，有未来函数，不建议用
    def stockpool_index(self, context, index, pool_id=1):
        log.info(self.name, '--stockpool_index获取指数成分股函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 获取指数成份股
        lists = list(get_index_stocks(index))
        # ·如果pool_id为0,则直接返回原始的成分股列表。
        if pool_id == 0:
            pass
        # ·如果pool_id为1,则进行进一步的筛选：
        # 。过滤掉创业板（股票代码以'30'开头）、科创板（股票代码以'68'开头）、北交所（股票代码以'8'或'4'开头）的股票。
        # 。过滤掉停牌（paused)、ST(is_st)、当日涨停（day_open等于high_limit)、当日跌停（day_open等于low_limit)的股票。
        # 。过滤掉名称中包含'ST'、"*'、'退'的股票。
        # 返回筛选后的股票列表：将经过筛选的股票列表返回。
        elif pool_id == 1:
            # 过滤创业板、ST、停牌、当日涨停
            # TODO
            current_data = get_current_data()
            # 经过测试，这里可以拿到未来的价格
            # log.error('605179.XSHG', current_data['605179.XSHG'].day_open, '--', current_data['605179.XSHG'].high_limit)
            # log.error('603833.XSHG', current_data['603833.XSHG'].day_open, '--', current_data['603833.XSHG'].high_limit)
            lists = [stock for stock in lists if not (
                    (current_data[stock].day_open == current_data[stock].high_limit) or  # 涨停开盘
                    (current_data[stock].day_open == current_data[stock].low_limit) or  # 跌停开盘
                    current_data[stock].paused or  # 停牌
                    current_data[stock].is_st or  # ST
                    ('ST' in current_data[stock].name) or
                    ('*' in current_data[stock].name) or
                    ('退' in current_data[stock].name) or
                    (stock.startswith('30')) or  # 创业
                    (stock.startswith('68')) or  # 科创
                    (stock.startswith('8')) or  # 北交
                    (stock.startswith('4'))  # 北交
            )
                     ]

        return lists

    # 选股
    def select(self, context):
        log.info(self.name, '--select函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 空仓期控制
        if self.use_empty_month and context.current_dt.month in (self.empty_month):
            return
        # 止损期控制
        if self.stoplost_date is not None:
            return
        select.select_list = []

    # 打印交易计划
    def print_trade_plan(self, context, select_list):
        now = str(context.current_dt.date()) + ' ' + str(context.current_dt.time())
        log.info(self.name, '--print_trade_plan函数--', now)

        # 1.获取子投资组合信息：从context中获取当前的子投资组合subportfolio,以及子投资组合的索引 self.subportfolio_index
        subportfolio = context.subportfolios[self.subportfolio_index]
        positions = subportfolio.long_positions
        positions_count = len(positions)
        current_data = get_current_data()  # 取股票名称

        content = now + ' ' + self.name + " 交易计划：" + "\n"

        # 仓位可用余额
        value_amount = subportfolio.available_cash
        # 遍历当前持仓的股票列表 subportfolio.long_positions,如果某只股票不在选股列表select_list的前self.max_hold_count只股票中，则将其标记为卖出。

        # 实时过滤部分股票，否则也买不了，放出去也没有意义
        target_list = self.utilstool.filter_lowlimit_stock(context, self.select_list)
        target_list = self.utilstool.filter_highlimit_stock(context, target_list)
        target_list = self.utilstool.filter_paused_stock(context, target_list)
        # 股票卖出的条件
        # 1. 有持仓
        # 2. 在目标列表中--不卖
        # 3. 不在目标列表中
        #     涨停：不卖
        #     不涨停：卖
        for stock in positions:
            if stock not in target_list[:self.max_hold_count] and stock not in self.yestoday_high_limit_list:
                last_prices = history(1, unit='1m', field='close', security_list=stock)
                current_data = get_current_data()
                if last_prices[stock][-1] < current_data[stock].high_limit:
                    content = content + stock + ' ' + current_data[stock].name + ' 未涨停卖出-- ' + str(
                        positions[stock].value) + '\n<br> '
                    value_amount = value_amount + positions[stock].value
                    positions_count = positions_count - 1

        # 计算买入金额
        # 如果买入数量buy_count大于0,则将可用现金除以买入数量，得到每只股票的买入金额。
        if len(target_list) > self.max_hold_count:
            buy_count = self.max_hold_count - positions_count
        else:
            buy_count = len(target_list) - positions_count
        if buy_count > 0:
            value_amount = value_amount / buy_count

        # 遍历选股列表
        # 如果某只股票不在当前持仓中，且在选股列表的前 self.max_hold_count只股票中，则将其标记为买入，并添加买入金额
        # 如果某只股票在当前持仓中，且在选股列表的前self.max_hold_count只股票中，则将其标记为继续持有。
        for stock in select_list:
            if stock not in subportfolio.long_positions and stock in select_list[:self.max_hold_count]:
                content = content + stock + ' ' + current_data[
                    stock].name + ' 买入-- ' + str(
                    value_amount) + '\n<br>'
            elif stock in subportfolio.long_positions and stock in select_list[:self.max_hold_count]:
                content = content + stock + ' ' + current_data[stock].name + ' 继续持有 \n<br>'
            else:
                # 兜底逻辑，一般用不到
                content = content + stock + ' ' + current_data[stock].name + '  持仓已满，备选股票 \n<br>'

        if ('买' in content) or ('持有' in content) or ('卖' in content):
            # weixin消息
            send_message(content)
            method_name = inspect.getframeinfo(inspect.currentframe()).function
            item = f"分仓策略:{self.name}<br>-函数名称:{method_name}<br>-时间:{now}"
            self.utilstool.send_wx_message(context, item, content)
            log.info(content)

    ##################################  风控函数群 ##################################

    # 空仓期检查
    def check_empty_month(self, context):
        log.info(self.name, '--check_empty_month函数：空仓期检查--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        subportfolio = context.subportfolios[self.subportfolio_index]
        if self.use_empty_month and context.current_dt.month in (self.empty_month) and len(
                subportfolio.long_positions) > 0:
            content = context.current_dt.date().strftime(
                "%Y-%m-%d") + self.name + ': 进入空仓期' + "\n" + "当前持仓股票: " + "\n"
            for stock in subportfolio.long_positions:
                content = content + stock + "\n"
            log.info(content)

    # 进入空仓期清仓
    def close_for_empty_month(self, context):
        log.info(self.name, '--close_for_empty_month函数：如果在空仓期会清仓所有股票--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        subportfolio = context.subportfolios[self.subportfolio_index]
        if self.use_empty_month and context.current_dt.month in (self.empty_month) and len(
                subportfolio.long_positions) > 0:
            self.sell(context, list(subportfolio.long_positions))

    # 止损检查
    # 实现了一个止损检查功能，它会根据股票的跌幅来决定是否需要止损，并在需要止损时记录止损日期和打印止损的股票列表。
    def check_stoplost(self, context):
        log.info(self.name, '--check_stoplost函数:止损检查--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        subportfolio = context.subportfolios[self.subportfolio_index]
        if self.use_stoplost:
            if self.stoplost_date is None:
                # 获取持仓股票的昨日收盘价
                last_prices = history(1, unit='1m', field='close', security_list=subportfolio.long_positions)
                for stock in subportfolio.long_positions:
                    position = subportfolio.long_positions[stock]
                    # 如果股票跌幅超stoplost_level:20%
                    if (position.avg_cost - last_prices[stock][-1]) / position.avg_cost > self.stoplost_level:
                        # 止损日记录到self.stoplost_date中
                        self.stoplost_date = context.current_dt.date()
                        log.info(self.name + ': ' + '开始止损')
                        content = context.current_dt.date().strftime("%Y-%m-%d") + ' ' + self.name + ': 止损' + "\n"
                        for stock in subportfolio.long_positions:
                            content = content + stock + "\n"
                        log.info(content)
                        # 一旦有股票需要止损，就不需要继续检查其他股票了。
                        break
            else:  # 已经在清仓静默期,stoplost_silent_days天后退出静默期
                if (context.current_dt + datetime.timedelta(
                        days=-self.stoplost_silent_days)).date() >= self.stoplost_date:
                    self.stoplost_date = None
                    log.info(self.name + ': ' + '退出止损')

    # 止损时清仓
    def close_for_stoplost(self, context):
        log.info(self.name, '--close_for_stoplost函数：如果在止损期会清仓所有股票--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        subportfolio = context.subportfolios[self.subportfolio_index]
        if self.use_stoplost and self.stoplost_date is not None and len(subportfolio.long_positions) > 0:
            self.sell(context, list(subportfolio.long_positions))

    # 3-8 判断今天是否为账户资金再平衡的日期(暂无使用)
    # date_flag,1-单个月，2-两个月1和4，3-三个月1和4和6
    def today_is_between(self, context, date_flag, start_date, end_date):
        today = context.current_dt.strftime('%m-%d')
        # 1(01-01~01-31)-4(04-01~04-30)-6(06-01~06-30)
        if date_flag == 1:
            if (start_date <= today) and (today <= end_date):
                return True
            else:
                return False
        elif date_flag == 2:
            if ('01-01' <= today) and (today <= '01-31'):
                return True
            elif ('04-01' <= today) and (today <= '04-30'):
                return True
            else:
                return False
        elif date_flag == 2:
            if ('01-01' <= today) and (today <= '01-31'):
                return True
            elif ('04-01' <= today) and (today <= '04-30'):
                return True
            elif ('06-01' <= today) and (today <= '06-30'):
                return True
            else:
                return False

    ##################################  交易函数群 ##################################
    # 调仓
    def adjustwithnoRM(self, context, only_buy=False, only_sell=False, together=True, is_single_buy=False):
        log.info(self.name, '--adjustwithnoRM调仓函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 空仓期或者止损期不再进行调仓
        if self.use_empty_month and context.current_dt.month in (self.empty_month):
            log.info('adjustwithnoRM调仓函数不再执行，因为当前月份是空仓期，空仓期月份为：', self.empty_month)
            return
        # 止损期控制
        if self.stoplost_date is not None:
            log.info('adjustwithnoRM调仓函数不再执行，因为当前时刻还处于止损期，止损期从:', self.stoplost_date, '开始')
            return

        # 先卖后买
        hold_list = list(context.subportfolios[self.subportfolio_index].long_positions)
        # 售卖列表：不在select_list前max_hold_count中的股票都要被卖掉
        sell_stocks = []
        # 实时过滤部分股票，否则也买不了，放出去也没有意义
        target_list = self.utilstool.filter_lowlimit_stock(context, self.select_list)
        target_list = self.utilstool.filter_highlimit_stock(context, target_list)
        target_list = self.utilstool.filter_paused_stock(context, target_list)
        # 股票卖出的条件
        # 1. 有持仓
        # 2. 在目标列表中--不卖
        # 3. 不在目标列表中
        #     涨停：不卖
        #     不涨停：卖

        for stock in hold_list:
            if stock not in target_list[:self.max_hold_count] and stock not in self.yestoday_high_limit_list:
                last_prices = history(1, unit='1m', field='close', security_list=stock)
                current_data = get_current_data()
                if last_prices[stock][-1] < current_data[stock].high_limit:
                    sell_stocks.append(stock)

        if only_buy:
            self.buy(context, target_list, is_single_buy)
            return
        if only_sell:
            self.sell(context, sell_stocks)
            return
        if together:
            self.sell(context, sell_stocks)
            self.buy(context, target_list, is_single_buy)
            return

    def specialBuy(self, context):
        log.info(self.name, '--specialBuy调仓函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        # 实时过滤部分股票，否则也买不了，放出去也没有意义
        target_list = self.utilstool.filter_lowlimit_stock(context, self.select_list)
        target_list = self.utilstool.filter_highlimit_stock(context, target_list)
        target_list = self.utilstool.filter_paused_stock(context, target_list)

        # 持仓列表
        subportfolios = context.subportfolios[self.subportfolio_index]
        log.debug('当前持仓:', subportfolios.long_positions)
        if target_list:
            if subportfolios.long_positions:
                value = subportfolios.available_cash / len(target_list)
                for stock in target_list:
                    self.utilstool.open_position(context, stock, value)
            else:
                value = subportfolios.total_value * 0.5 / len(target_list)
                for stock in target_list:
                    self.utilstool.open_position(context, stock, value)

    def specialSell(self, context):
        log.info(self.name, '--SpecialSell调仓函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 持仓列表
        hold_positions = context.subportfolios[self.subportfolio_index].long_positions
        hold_list = list(hold_positions)
        # 售卖列表：不在select_list前max_hold_count中的股票都要被卖掉
        sell_stocks = []
        for stock in hold_list:
            current_data = get_current_data()
            MA5 = MA(stock, check_date=context.current_dt, timeperiod=5)  #
            position = hold_positions[stock]

            # 有可卖出的仓位  &  当前股票没有涨停 & 当前的价格大于持仓价（有收益）
            if ((position.closeable_amount != 0) and (
                    current_data[stock].last_price < current_data[stock].high_limit) and (
                    current_data[stock].last_price > 1 * position.avg_cost)):  # avg_cost当前持仓成本
                sell_stocks.append(stock)
            # 有可卖出的仓位  &  跌破5日线止损
            if ((position.closeable_amount != 0) and (current_data[stock].last_price < MA5[stock])):
                sell_stocks.append(stock)

        self.sell(context, sell_stocks)

    # 涨停打开卖出
    def sell_when_highlimit_open(self, context):
        log.info(self.name, '--sell_when_highlimit_open涨停打开卖出股票函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        if self.yestoday_high_limit_list != []:
            for stock in self.yestoday_high_limit_list:
                if stock in context.subportfolios[self.subportfolio_index].long_positions:
                    current_data = get_price(stock, end_date=context.current_dt, frequency='1m',
                                             fields=['close', 'high_limit'],
                                             skip_paused=False, fq='pre', count=1, panel=False, fill_paused=True)
                    if current_data.iloc[0, 0] < current_data.iloc[0, 1]:
                        self.sell(context, [stock])
                        self.is_stoplost_or_highlimit = True
                        content = context.current_dt.date().strftime(
                            "%Y-%m-%d") + ' ' + self.name + ': {}涨停打开，卖出'.format(stock) + "\n"
                        log.info(content)

    # 买入多只股票
    def buy(self, context, buy_stocks, is_single_buy=False):

        log.info(self.name, '--buy函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        subportfolio = context.subportfolios[self.subportfolio_index]
        if is_single_buy and len(subportfolio.long_positions) > 0:
            # 如果有持仓，还有选票就先不买了
            buy_count = 0
        elif len(buy_stocks) > self.max_hold_count:
            buy_count = self.max_hold_count - len(subportfolio.long_positions)
        else:
            buy_count = len(buy_stocks) - len(subportfolio.long_positions)
        if buy_count > 0:
            value = subportfolio.available_cash / buy_count
            index = 0
            for stock in buy_stocks:
                if stock in subportfolio.long_positions:
                    continue
                # 如果买入失败，顺延买下一个
                if self.utilstool.open_position(context, stock, value):
                    index = index + 1
                    if index >= buy_count:
                        break

    # 卖出多只股票
    def sell(self, context, sell_stocks):

        log.info(self.name, '--sell函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        subportfolio = context.subportfolios[self.subportfolio_index]
        for stock in sell_stocks:
            if stock in subportfolio.long_positions:
                self.utilstool.close_position(context, stock, 0)

    # 计算夏普系数的函数
    def cal_sharpe_ratio(self, returns, rf, type):  # portfolio_daily_returns 是一个包含每日收益的列表
        annual_periods = 250  # 假设一年有250个交易日
        sharpe_ratio = 0
        if (type == 'MEAN'):
            returns = returns - rf / annual_periods  # 计算超额收益
            return_mean = np.mean(returns) * annual_periods  # 简单年化收益率 = 投资组合的平均超额收益率 * 年化期数
            std_annualized = returns.std() * np.sqrt(annual_periods)  # 计算年化标准差
            if std_annualized == 0:  # 计算夏普比率
                sharpe_ratio = 0
            else:
                sharpe_ratio = return_mean / std_annualized
        if (type == 'CAGR'):
            returns = returns - rf / annual_periods  # 计算超额收益
            years = len(returns) / annual_periods  # 投资期数
            total = returns.add(1).prod() - 1  # 计算年化收益率
            return_annualized = (total + 1.0) ** (1.0 / years) - 1  # 年化收益率
            std_annualized = returns.std() * np.sqrt(annual_periods)  # 计算年化标准差
            if std_annualized == 0:  # 计算夏普比率
                sharpe_ratio = 0
            else:
                sharpe_ratio = return_annualized / std_annualized
        return sharpe_ratio

    ## 收盘后运行函数
    def after_market_close(self, context):
        now = str(context.current_dt.date()) + ' ' + str(context.current_dt.time())
        log.info(self.name, '--after_market_close收盘后运行函数--', now)

        subportfolio = context.subportfolios[self.subportfolio_index]

        # 计算当前盈利
        title = self.name + '收益率'
        # subportfolio_startcash=context.portfolio.starting_cash*g.portfolio_value_proportion[self.subportfolio_index]+subportfolio.inout_cash
        # 账户累计出入金
        subportfolio_startcash = subportfolio.inout_cash
        if subportfolio_startcash != 0:
            ret_ratio = round((subportfolio.total_value / subportfolio_startcash - 1), 2)
        else:
            ret_ratio = 0

        kv = {title: ret_ratio}
        record(**kv)
        orders = get_orders()
        trades = get_trades()
        # 创建一个 prettytable 对象,打印当天交易信息
        trade_table = PrettyTable(
            ["策略名称", "代码", "证券名称", "交易方向", "交易时间", "交易数量", "交易价格", "盈亏情况"])
        transaction = 0

        if len(trades) > 0:
            for _trade in trades.values():
                if (self.subportfolio_index == orders[_trade.order_id].pindex):
                    transaction += 1
                    # strategy_index = orders[_trade.order_id].pindex
                    strategy_name = self.name
                    security = _trade.security[:20]
                    name = get_security_info(_trade.security).display_name
                    action = '买入' if orders[_trade.order_id].is_buy else '卖出'
                    if orders[_trade.order_id].is_buy == False:
                        # 卖出的时候可以计算收益情况
                        self.trade_num += 1
                        if _trade.price > round(orders[_trade.order_id].avg_cost, 2):
                            # print('交易日志：',name, _trade.price, round(orders[_trade.order_id].avg_cost,2))
                            self.win_num += 1
                        self.win_lose_rate = self.win_num / self.trade_num
                    # print(self.trade_num,self.win_num,self.win_lose_rate)
                    tradedate = _trade.time
                    tradeamount = _trade.amount
                    tradeprice = _trade.price
                    profit_percent_trade = (_trade.price / orders[_trade.order_id].avg_cost - 1) * 100
                    trade_table.add_row(
                        [strategy_name, security, name, action, tradedate, tradeamount, f"{tradeprice:.3f}",
                         f"{profit_percent_trade:.3f}%"])

        method_name = inspect.getframeinfo(inspect.currentframe()).function
        item = f"分仓策略:{self.name}<br>-函数名称:{method_name}<br>-时间:{now}"
        content_log = ''
        content_wx = ''
        if transaction > 0:
            content_wx = content_wx + '#############<br><br><br>' + f"{self.name} 策略当日交易信息: <br>{self.utilstool.pretty_table_to_kv_string(trade_table)}<br>"
            content_log = content_log + '#############\n\n\n' + f"{self.name} 策略当日交易信息: \n{trade_table}\n"

            # write_file(g.logfile,f'\n{trade_table}', append=True)
            # pass
        else:
            content_log = content_log + '#############' + self.name + '当天没有任何交易#############\n'
            content_wx = content_wx + '#############' + self.name + '当天没有任何交易#############<br>'

            # write_file(g.logfile,'-'*20+self.name+'当天没有任何交易'+'-'*20+'\n', append=True)
            # pass

        # 创建一个 prettytable 对象,打印当天持仓信息
        pos_table = PrettyTable(
            ["策略名称", "代码", "证券名称", "买入日期", "买入价格", "现价", "收益率", "持股数", "市值"])
        if len(list(subportfolio.long_positions)) > 0:
            for stock in list(subportfolio.long_positions):
                position = subportfolio.long_positions[stock]
                security = position.security[:20]
                name = get_security_info(position.security).display_name
                buyindate = position.init_time.date()
                buyinprice = position.avg_cost
                currprice = position.price
                # 股票收益率
                profit_percent_hold = (position.price / position.avg_cost - 1) * 100
                # 股票价值
                value = position.value / 10000
                # 股票持股数
                amount = position.total_amount
                pos_table.add_row([self.name, security, name, buyindate, f"{buyinprice:.3f}", f"{currprice:.3f}",
                                   f"{profit_percent_hold:.3f}%", amount, f"{value:.3f}万"])
            # print(f'\n{pos_table}')

            content_wx = content_wx + "#############<br><br><br>" + f"{self.name} 策略当日持仓信息: <br>{self.utilstool.pretty_table_to_kv_string(pos_table)}<br>"
            content_log = content_log + "#############\n\n\n" + f"{self.name} 策略当日持仓信息: \n{pos_table}\n"

            # write_file(g.logfile,f'\n{pos_table}', append=True)
        else:
            content_wx = content_log + '#############' + self.name + '当天没有持仓#############<br>'
            content_log = content_log + '#############' + self.name + '当天没有持仓#############\n'

            # write_file(g.logfile,'-'*20+self.name+'当天没有任何交易'+'-'*20+'\n', append=True)
            # pass

        # 创建一个 prettytable 对象,打印当天账户信息
        account_table = PrettyTable(
            ["日期", "策略名称", "策略总资产", "策略持仓总市值", "策略可用现金", "策略当天出入金", "策略当天收益率",
             "策略累计收益率", "策略胜率", "策略夏普比率", "策略最大回撤", "最大回撤区间"])
        date = str(context.current_dt.date()) + ' ' + str(context.current_dt.time())
        # 账户可用现金
        cash = subportfolio.available_cash / 10000
        # 账户持仓价值
        pos_value = subportfolio.positions_value / 10000
        total_assets = subportfolio.total_value / 10000
        new_data = {'date': date, 'total_value': subportfolio.total_value}
        self.portfolio_value = self.portfolio_value.append(new_data, ignore_index=True)
        # 计算当日之前的资金曲线最高点
        self.portfolio_value['max2here'] = self.portfolio_value['total_value'].expanding().max()
        # 计算历史最高值到当日的剩余量drawdown
        self.portfolio_value['dd2here'] = self.portfolio_value['total_value'] / self.portfolio_value['max2here']
        # 计算回撤完之后剩余量的最小值(也就是最大回撤的剩余量)，以及最大回撤的结束时间
        end_date, remains = tuple(self.portfolio_value.sort_values(by=['dd2here']).iloc[0][['date', 'dd2here']])
        # 计算最大回撤开始时间
        start_date = self.portfolio_value[self.portfolio_value['date'] <= end_date].sort_values(by='total_value',
                                                                                                ascending=False).iloc[
            0]['date']
        max_draw_down = (1 - remains) * 100
        daily_returns = self.portfolio_value['total_value'].pct_change()

        if (self.inout_cash != 0):
            daily_returns.iloc[-1] = (self.portfolio_value['total_value'].iloc[-1] - self.inout_cash) / \
                                     self.portfolio_value['total_value'].iloc[-2] - 1

        self.sharp = self.cal_sharpe_ratio(daily_returns, rf=0.04, type='CAGR')
        if subportfolio_startcash != 0:
            total_return = subportfolio.total_value / subportfolio_startcash - 1
        else:
            total_return = 0
        account_table.add_row([date, self.name, f"{total_assets:.3f}万", f"{pos_value:.3f}万", f"{cash:.3f}万",
                               f"{self.inout_cash / 10000:.3f}万", f"{daily_returns.iloc[-1] * 100:.3f}%",
                               f"{total_return * 100:.3f}%", f"{self.win_lose_rate:.3f}", f"{self.sharp:.3f}",
                               f"{max_draw_down:.3f}%", f"{start_date}到{end_date}"])
        self.previous_portfolio_value = subportfolio.total_value

        content_wx = content_wx + "#############<br><br><br>" + f"{self.name} 策略当日账户信息: <br>{self.utilstool.pretty_table_to_kv_string(account_table)}<br>"
        content_log = content_log + "#############\n\n\n" + f"{self.name} 策略当日账户信息: \n{account_table}\n"

        # write_file(g.logfile,f'\n{account_table}', append=True)

        log.info(content_log)
        self.utilstool.send_wx_message(context, item, content_wx)
        log.info('-------------分割线-------------')
        # write_file(g.logfile,'-'*20+date+'日志终结'+'-'*20+'\n'+'\n', append=True)
        self.inout_cash = 0

    def clear_append_buy_dict(self, context):  # 卖出补跌的仓位
        now = str(context.current_dt.date()) + ' ' + str(context.current_dt.time())
        log.info(self.name, '--clear_append_buy_dict函数--', now)

        if self.bought_stocks:
            for stock, amount in self.bought_stocks.items():
                positions = context.subportfolios[self.subportfolio_index].long_positions
                if stock in positions:
                    self.utilstool.close_position(context, stock, -amount, False)
                # 清空记录
            self.bought_stocks.clear()

    def append_buy_dict(self, context):
        now = str(context.current_dt.date()) + ' ' + str(context.current_dt.time())
        log.info(self.name, '--append_buy_dict 补买函数--', now)
        subportfolios = context.subportfolios[self.subportfolio_index]
        positions = subportfolios.long_positions

        append_buy_dict = {}
        for stock in self.hold_list:
            if stock in positions:
                position = positions[stock]
                current_price = position.price
                avg_cost = position.avg_cost

                if current_price < avg_cost * 0.92:
                    log.info("止损 Selling out %s" % (stock))
                    self.sell(context, [stock])
                    self.is_stoplost_or_highlimit = True
                else:
                    rate = (current_price - avg_cost) / avg_cost
                    append_buy_dict[stock] = rate
        if self.is_stoplost_or_highlimit and append_buy_dict:
            self.is_stoplost_or_highlimit = False
            # 清空记录
            num = 3
            sorted_items = sorted(append_buy_dict.items(), key=lambda x: x[1])  # 按照值进行排序，返回包含(key, value)元组的列表
            result_stock = [item[0] for item in sorted_items[:num]]  # 取前N个元组中的key

            cash = subportfolios.available_cash / num
            log.info("补跌最多的3支 股票代码: %s" % result_stock)
            for stock in result_stock:
                self.utilstool.open_position(context, stock, cash, False)
                if stock not in self.bought_stocks:
                    self.bought_stocks[stock] = cash
