'''
多策略分子账户并行
用到的策略：
蚂蚁量化,东哥：白马股攻防转换策略（BMZH策略）
linlin2018，ZLH：低波全天候策略（外盘ETF策略）
@荒唐的方糖大佬:国九条小市值（XSZGJT）（还可以改进）
'''

# 导入函数库
# -*- coding: utf-8 -*-
# 如果你的文件包含中文, 请在文件的第一行使用上面的语句指定你的文件编码

# 用到策略及数据相关API请加入下面的语句(如果要兼容研究使用可以使用 try except导入
from kuanke.user_space_api import *
from jqdata import *
from jqfactor import get_factor_values
from kuanke.wizard import *
import numpy as np
import pandas as pd
import talib
import datetime as dt
import math
import talib as tl
from jqlib.technical_analysis import *
from scipy.linalg import inv
import pickle
import requests
from prettytable import PrettyTable
import inspect
from collections import OrderedDict


# 初始化函数，设定基准等等
def initialize(context):
    log.warn('--initialize函数(只运行一次)--',
             str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
    # 设定沪深300作为基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    # 过滤掉order系列API产生的比error级别低的log
    log.set_level('order', 'error')
    # 关闭未来函数
    set_option('avoid_future_data', True)

    ### 股票相关设定 ###
    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.0005, open_commission=0.0001, close_commission=0.0001, min_commission=0),
                   type='stock')

    # 为股票设定滑点为百分比滑点
    set_slippage(PriceRelatedSlippage(0.002), type='stock')

    # 临时变量

    # 持久变量
    g.strategys = {}
    # 子账户 分仓
    g.portfolio_value_proportion = [0.35, 0.15, 0.50]

    # 创建策略实例
    # 初始化策略子账户 subportfolios
    set_subportfolios([
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[0], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[1], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[2], 'stock'),
    ])

    context.subportfolios_name_map = {
        0: '白马策略',
        1: 'ETF策略',
        2: '小市值策略'
    }
    # 是否发送微信消息，回测环境不发送，模拟环境发送
    context.is_send_wx_message = 0
    params = {
        'max_hold_count': 1,  # 最大持股数
        'max_select_count': 3,  # 最大输出选股数
    }
    # 白马策略，第一个仓
    bmzh_strategy = BMZH_Strategy(context, subportfolio_index=0, name='白马股攻防转换策略', params=params)
    g.strategys[bmzh_strategy.name] = bmzh_strategy

    params = {
        'max_hold_count': 1,  # 最大持股数
        'max_select_count': 2,  # 最大输出选股数
    }
    # ETF 策略，第二个仓
    wpetf_strategy = WPETF_Strategy(context, subportfolio_index=1, name='外盘ETF轮动策略', params=params)
    g.strategys[wpetf_strategy.name] = wpetf_strategy

    params = {
        'max_hold_count': 3,  # 最大持股数
        'max_select_count': 10,  # 最大输出选股数
        'use_empty_month': True,  # 是否在指定月份空仓
        'empty_month': [1, 4],  # 指定空仓的月份列表
        'use_wide_time_strategies' : True # 使用宽度择时
    }
    # 小世值，第三个仓
    xszgjt_strategy = XSZ_GJT_Strategy(context, subportfolio_index=2, name='国九条小市值策略', params=params)
    g.strategys[xszgjt_strategy.name] = xszgjt_strategy


# 模拟盘在每天的交易时间结束后会休眠，第二天开盘时会恢复，如果在恢复时发现代码已经发生了修改，则会在恢复时执行这个函数。 具体的使用场景：可以利用这个函数修改一些模拟盘的数据。
def after_code_changed(context):  # 输出运行时间
    log.info('函数运行时间(after_code_changed)：' + str(context.current_dt.time()))

    # 是否发送微信消息，回测环境不发送，模拟环境发送
    context.is_send_wx_message = 0

    unschedule_all()  # 取消所有定时运行

    run_daily(check_width_timing, "14:50")  # 宽度清仓

    # 执行计划
    # 选股函数--Select：白马和 ETF 分开使用
    # 执行函数--adjust：白马和 ETF 轮动共用一个
    # # 白马，按月运行 TODO
    if g.portfolio_value_proportion[0] > 0:
        run_monthly(bmzh_select, 1, time='7:40')  # 阅读完成，测试完成
        run_monthly(bmzh_adjust, 1, time='09:35')  # 阅读完成，测试完成
        run_daily(bmzh_after_market_close, 'after_close')
    #
    # # ETF轮动，按天运行
    if g.portfolio_value_proportion[1] > 0:
        run_daily(wpetf_select, time='7:42')  # 阅读完成，测试完成
        run_daily(wpetf_adjust, time='09:35')  # 阅读完成，测试完成
        run_daily(wpetf_after_market_close, 'after_close')

    # # 小市值，按天/周运行
    if g.portfolio_value_proportion[2] > 0:
        run_daily(xszgjt_day_prepare, time='7:33')
        run_weekly(xszgjt_select, 2, time='7:43')
        run_daily(xszgjt_open_market, time='9:30')
        run_weekly(xszgjt_adjust, 2, time='9:35')
        # run_daily(xszgjt_sell_when_highlimit_open, time='11:27')
        run_daily(xszgjt_sell_when_highlimit_open, time='11:20')
        run_daily(xszgjt_sell_when_highlimit_open, time='14:50')
        # run_daily(xszgjt_append_buy_stock, time='14:51')
        run_daily(xszgjt_after_market_close, 'after_close')
        # run_daily(xszgjt_print_position_info, time='15:10')

# 中央数据管理器==========================================================
class DataManager:
    def __init__(self, max_cache_size=100):
        self.cache = OrderedDict()
        self.max_cache_size = max_cache_size

    def get_data(self, key, func, *args, **kwargs):
        if key not in self.cache:
            if len(self.cache) >= self.max_cache_size:
                self.cache.popitem(last=False)
            self.cache[key] = func(*args, **kwargs)
        self.cache.move_to_end(key)
        return self.cache[key]


data_manager = DataManager()

def check_width_timing(context):
    for strategy_name, strategy in g.strategys.items():
        if strategy.use_wide_time_strategies:
            log.info(strategy_name, '--check_width_timing函数--',
                     str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
            # 获取指数成分股
            stocks = data_manager.get_data(
                f"index_stocks_000985_{context.previous_date.strftime('%Y-%m')}",
                get_index_stocks,
                "000985.XSHG"
            )
            # 获取前一交易日及前6个交易日的价格数据
            prices = data_manager.get_data(
                f"market_breadth_prices_{stocks}_{context.previous_date.strftime('%Y-%m-%d')}",
                get_price,
                stocks, end_date=context.previous_date, frequency="1d", fields=["close"], count=7, panel=False
            )
            prices["date"] = pd.DatetimeIndex(prices.time).date
            close_pivot = prices.pivot(index="code", columns="date", values="close").dropna(axis=0)
            # 计算前5个交易日宽度总分平均值
            prev_5_days_close_pivot = close_pivot.iloc[:, -6:-1]
            prev_5_days_ma5 = prev_5_days_close_pivot.rolling(window=5, axis=1).mean().iloc[:, -1:]
            prev_5_days_bias = prev_5_days_close_pivot.iloc[:, -1:] > prev_5_days_ma5
            prev_5_days_daily_scores = (prev_5_days_bias.sum() * 100.0 / prev_5_days_bias.count()).round()
            prev_5_days_avg_score = prev_5_days_daily_scores.iloc[-1]

            # 计算当天的宽度分（使用实时数据）
            current_data = get_current_data()
            current_prices = pd.Series(
                {stock: current_data[stock].last_price for stock in stocks if stock in current_data})
            current_ma5 = close_pivot.iloc[:, -5:].mean(axis=1)
            valid_stocks = current_prices.index.intersection(current_ma5.index)
            current_prices = current_prices[valid_stocks]
            current_ma5 = current_ma5[valid_stocks]
            current_bias = current_prices > current_ma5

            # 检查分母是否为零
            if current_bias.count() == 0:
                log.warning("current_bias.count() is zero, skipping current_score calculation.")
                continue

            current_score = (current_bias.sum() * 100.0 / current_bias.count()).round()

            # 判断是否需要空仓
            if current_score < prev_5_days_avg_score * 0.8:
                index = int(strategy.subportfolio_index)
                subportfolio = context.subportfolios[index]
                for stock in list(subportfolio.long_positions.keys()):
                    strategy.utilstool.close_position(context, stock, 0, True)

# 选股
def bmzh_select(context):
    g.strategys['白马股攻防转换策略'].select(context)


# 交易
def bmzh_adjust(context):
    g.strategys['白马股攻防转换策略'].adjustwithnoRM(context)


# 收盘统计
def bmzh_after_market_close(context):
    g.strategys['白马股攻防转换策略'].after_market_close(context)


def wpetf_select(context):
    g.strategys['外盘ETF轮动策略'].select(context)


def wpetf_adjust(context):
    g.strategys['外盘ETF轮动策略'].adjustwithnoRM(context)


def wpetf_after_market_close(context):
    g.strategys['外盘ETF轮动策略'].after_market_close(context)


def xszgjt_day_prepare(context):
    g.strategys['国九条小市值策略'].day_prepare(context)


def xszgjt_select(context):
    g.strategys['国九条小市值策略'].select(context)


def xszgjt_adjust(context):
    g.strategys['国九条小市值策略'].clear_append_buy_dict(context)
    g.strategys['国九条小市值策略'].adjustwithnoRM(context)


def xszgjt_open_market(context):
    g.strategys['国九条小市值策略'].close_for_empty_month(context)
    g.strategys['国九条小市值策略'].close_for_stoplost(context)


def xszgjt_sell_when_highlimit_open(context):
    g.strategys['国九条小市值策略'].sell_when_highlimit_open(context)


def xszgjt_append_buy_stock(context):
    g.strategys['国九条小市值策略'].append_buy_dict(context)


def xszgjt_after_market_close(context):
    g.strategys['国九条小市值策略'].after_market_close(context)



class UtilsToolClass:
    def __init__(self):
        self.name = None
        self.subportfolio_index = None

    def set_params(self, name, subportfolio_index):
        self.name = name
        self.subportfolio_index = subportfolio_index

    # 计算左压天数
    def calculate_zyts(self, context, stock):
        high_prices = attribute_history(stock, 101, '1d', fields=['high'], skip_paused=True)['high']
        prev_high = high_prices.iloc[-1]
        zyts_0 = next((i - 1 for i, high in enumerate(high_prices[-3::-1], 2) if high >= prev_high), 100)
        zyts = zyts_0 + 5
        return zyts

    def transform_date(self, context, date, date_type):
        if type(date) == str:
            str_date = date
            dt_date = dt.datetime.strptime(date, '%Y-%m-%d')
            d_date = dt_date.date()
        elif type(date) == dt.datetime:
            str_date = date.strftime('%Y-%m-%d')
            dt_date = date
            d_date = dt_date.date()
        elif type(date) == dt.date:
            str_date = date.strftime('%Y-%m-%d')
            dt_date = dt.datetime.strptime(str_date, '%Y-%m-%d')
            d_date = date
        dct = {'str': str_date, 'dt': dt_date, 'd': d_date}
        return dct[date_type]

    def get_shifted_date(self, context, date, days, days_type='T'):
        # 获取上一个自然日
        d_date = self.transform_date(context, date, 'd')
        yesterday = d_date + dt.timedelta(-1)
        # 移动days个自然日
        if days_type == 'N':
            shifted_date = yesterday + dt.timedelta(days + 1)
        # 移动days个交易日
        if days_type == 'T':
            all_trade_days = [i.strftime('%Y-%m-%d') for i in list(get_all_trade_days())]
            # 如果上一个自然日是交易日，根据其在交易日列表中的index计算平移后的交易日
            if str(yesterday) in all_trade_days:
                shifted_date = all_trade_days[all_trade_days.index(str(yesterday)) + days + 1]
            # 否则，从上一个自然日向前数，先找到最近一个交易日，再开始平移
            else:
                for i in range(100):
                    last_trade_date = yesterday - dt.timedelta(i)
                    if str(last_trade_date) in all_trade_days:
                        shifted_date = all_trade_days[all_trade_days.index(str(last_trade_date)) + days + 1]
                        break
        return str(shifted_date)

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
                lists = self.filter_kcbj_stock(context, lists)
            if is_filter_st:
                lists = self.filter_st_stock(context, lists)
            if is_filter_paused:
                lists = self.filter_paused_stock(context, lists)
            if is_filter_highlimit:
                lists = self.filter_highlimit_stock(context, lists)
            if is_filter_lowlimit:
                lists = self.filter_lowlimit_stock(context, lists)
            if is_filter_new:
                lists = self.filter_new_stock(context, lists, days=375)

        return lists

    ##################################  交易函数群 ##################################

    # 开仓单只
    def open_position(self, context, security, value, target=True):
        now = str(context.current_dt.date()) + ' ' + str(context.current_dt.time())
        now_time = context.current_dt.time()
        current_data = get_current_data()
        before_buy = dt.time(9, 30) > now_time
        # log.info('before_buy:',before_buy)
        style_arg = MarketOrderStyle(current_data[security].day_open) if before_buy else None
        if target:
            order_info = order_target_value(security, value, style=style_arg, pindex=self.subportfolio_index)
        else:
            # log.info('S:', security, "--value:", value)
            order_info = order_value(security, value, style=style_arg, pindex=self.subportfolio_index)

        method_name = inspect.getframeinfo(inspect.currentframe()).function
        item = f"分仓策略:{self.name}<br>-函数名称:{method_name}<br>-时间:{now}"
        if order_info != None and order_info.filled > 0:
            content = (f"策略: {self.name} "
                       f"--操作时间: {now} "
                       f"--买入股票: {security} "
                       f"--计划买入金额: {value} "
                       f"--买入数量: {order_info.amount} "
                       f"--成交数量: {order_info.filled} "
                       f"--买入均价: {order_info.price} "
                       f"--实际买入金额: {order_info.price * order_info.filled} "
                       f"--交易佣金: {order_info.commission:.2f}\n<br>")
            log.info(content)
            send_message(content)
            self.send_wx_message(context, item, content)
            return True
        content = (f"策略: {self.name} "
                   f"--操作时间: {now} "
                   f"--买入股票，交易失败！！股票: {security} "
                   f"--失败原因: {order_info} "
                   f"--计划买入金额: {value}\n<br>")
        log.error(content)
        send_message(content)
        self.send_wx_message(context, item, content)
        return False

    # 清仓单只
    def close_position(self, context, security, value, target=True):
        now = str(context.current_dt.date()) + ' ' + str(context.current_dt.time())
        if target:
            order_info = order_target_value(security, value, pindex=self.subportfolio_index)
        else:
            order_info = order_value(security, value, pindex=self.subportfolio_index)
        method_name = inspect.getframeinfo(inspect.currentframe()).function
        item = f"分仓策略:{self.name}<br>-函数名称:{method_name}<br>-时间:{now}"

        if order_info != None and order_info.status == OrderStatus.held and order_info.filled == order_info.amount:
            # 计算收益率:（当前价格/持仓价格）- 1
            ret = 100 * (order_info.price / order_info.avg_cost - 1)
            # 计算收益金额: 可卖仓位 *（当前价格/持仓价格)
            ret_money = order_info.amount * (order_info.price - order_info.avg_cost)
            content = (f"策略: {self.name} "
                       f"--操作时间: {now} "
                       f"--卖出股票: {security} "
                       f"--卖出数量: {order_info.amount} "
                       f"--成交数量: {order_info.filled} "
                       f"--持仓均价: {order_info.avg_cost} "
                       f"--卖出均价: {order_info.price} "
                       f"--实际卖出金额: {order_info.price * order_info.filled} "
                       f"--交易佣金: {order_info.commission:.2f} 收益率: {ret:.2f}% 收益金额: {ret_money:.2f} \n<br>")
            log.info(content)
            send_message(content)
            self.send_wx_message(context, item, content)
            return True
        content = (f"策略: {self.name} "
                   f"--操作时间: {now} "
                   f"--失败原因: {order_info} "
                   f"--卖出股票，交易失败！！！股票: {security} \n<br>")
        log.error(content)
        send_message(content)
        self.send_wx_message(context, item, content)
        return False

    ##################################  选股函数群 ##################################

    # 获取股票股票池（暂无使用）
    def get_security_universe(self, context, security_universe_index, security_universe_user_securities):
        log.info(self.name, '--get_security_universe函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        temp_index = []
        for s in security_universe_index:
            if s == 'all_a_securities':
                temp_index += list(get_all_securities(['stock'], context.current_dt.date()).index)
            else:
                temp_index += get_index_stocks(s)
        for x in security_universe_user_securities:
            temp_index += x
        return sorted(list(set(temp_index)))

    # 过滤科创北交
    def filter_basic_stock(self, context, stock_list):
        log.info(self.name, '--filter_basic_stock过滤股票函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        current_data = get_current_data()
        return [
            stock
            for stock in stock_list
            if not current_data[stock].paused
               and not current_data[stock].is_st
               and "ST" not in current_data[stock].name
               and "*" not in current_data[stock].name
               and "退" not in current_data[stock].name
               and not (
                    stock.startswith('4') or
                    stock.startswith('8') or
                    stock.startswith('68') or
                    stock.startswith('30')
            )
               and not context.previous_date - get_security_info(stock).start_date < dt.timedelta(days=375)

        ]
        return stock_list

    # 过滤科创北交
    def filter_kcbj_stock(self, context, stock_list):
        log.info(self.name, '--filter_kcbj_stock过滤科创北交函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 使用列表推导式过滤股票
        filtered_stock_list = [stock for stock in stock_list if not (stock.startswith('4') or
                                                                     stock.startswith('8') or
                                                                     stock.startswith('68') or
                                                                     stock.startswith('30'))]

        return filtered_stock_list

    # 过滤停牌股票
    def filter_paused_stock(self, context, stock_list):
        log.info(self.name, '--filter_paused_stock过滤停牌股票函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        current_data = get_current_data()
        return [stock for stock in stock_list if not current_data[stock].paused]

    # 过滤ST及其他具有退市标签的股票
    def filter_st_stock(self, context, stock_list):
        log.info(self.name, '--filter_st_stock过滤ST及其他具有退市标签的股票函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        current_data = get_current_data()
        return [stock for stock in stock_list
                if not current_data[stock].is_st
                and 'ST' not in current_data[stock].name
                and '*' not in current_data[stock].name
                and '退' not in current_data[stock].name]

    # 过滤涨停的股票
    def filter_highlimit_stock(self, context, stock_list):
        log.info(self.name, '--filter_highlimit_stock过滤涨停的股票函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        subportfolio = context.subportfolios[self.subportfolio_index]
        last_prices = history(1, unit='1m', field='close', security_list=stock_list)
        current_data = get_current_data()

        return [stock for stock in stock_list if stock in subportfolio.long_positions
                or last_prices[stock][-1] < current_data[stock].high_limit]

    # 过滤跌停的股票
    def filter_lowlimit_stock(self, context, stock_list):
        log.info(self.name, '--filter_lowlimit_stock过滤跌停的股票函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        subportfolio = context.subportfolios[self.subportfolio_index]
        last_prices = history(1, unit='1m', field='close', security_list=stock_list)
        current_data = get_current_data()

        return [stock for stock in stock_list if stock in subportfolio.long_positions
                or last_prices[stock][-1] > current_data[stock].low_limit]

    # 过滤次新股（小市值专用）
    def filter_new_stock(self, context, stock_list, days):
        log.info(self.name, '--filter_new_stock过滤次新股函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        return [stock for stock in stock_list if
                not context.previous_date - get_security_info(stock).start_date < dt.timedelta(days=days)]

    # 过滤大幅解禁（小市值专用）
    def filter_locked_shares(self, context, stock_list, days):
        log.info(self.name, '--filter_locked_shares过滤解禁股函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 获取指定日期区间内的限售解禁数据
        df = get_locked_shares(stock_list=stock_list, start_date=context.previous_date.strftime('%Y-%m-%d'),
                               forward_count=days)
        # 过滤出解禁数量占总股本的百分比超过 20% 的股票
        df = df[df['rate1'] > 0.2]
        filterlist = list(df['code'])
        # 从股票池中排除这些股票
        return [stock for stock in stock_list if stock not in filterlist]

    ###################################  公用函数群 ##################################
    # 获取个股行业,暂无使用
    def get_industry_name(self, i_Constituent_Stocks, value):
        return [k for k, v in i_Constituent_Stocks.items() if value in v]

    # 把prettytable对象转换成键值对字符串
    def pretty_table_to_kv_string(self, table):
        headers = table.field_names
        result = ""
        data_rows = table._rows  # 直接获取表格内部存储的数据行列表，避免格式干扰
        for row in data_rows:
            for header, cell in zip(headers, row):
                result += f"{header}: {cell}\n<br>"
            result += "\n<br>"
        return result.rstrip()

    # 发送微信消息
    def send_wx_message(self, context, item, message):
        if context.is_send_wx_message != 1:
            return
        url = "https://wxpusher.zjiecode.com/api/send/message"

        data = {
            "appToken": "AT_B7CVGazuAWXoqBoIlGAzlIwkunQuXIQM",
            "content": f"<h1>{item}</h1><br/><p style=\"color:red;\">{message}</p>",
            "summary": item,
            "contentType": 2,
            "topicIds": [
                36105
            ],
            "url": "https://wxpusher.zjiecode.com",
            "verifyPay": False,
            "verifyPayType": 0
        }
        response = requests.post(url, json=data)
        # 可以根据需要查看响应的状态码、内容等信息
        # print(response.status_code)
        # print(response.text)

    # 计算市场宽度
    def get_market_breadth(self, context, max_industry_cnt):
        log.info(self.name, '--get_market_breadth--计算市场宽度，选择偏离程度最高的行业--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        """
        计算市场宽度，选择偏离程度最高的行业
        """
        # 指定日期以防止未来数据
        yesterday = context.previous_date

        # 获取上证中小盘指数的成分股
        stocks = get_index_stocks("000985.XSHG")

        # 获取历史收盘价数据，包括20天移动平均所需的数据
        count = 1
        historical_prices = get_price(
            stocks,
            end_date=yesterday,
            frequency="1d",
            fields=["close"],
            count=count + 20,
            panel=False,
        )

        # 将时间字段转换为日期
        historical_prices["date"] = pd.DatetimeIndex(historical_prices['time']).date

        # 将数据重塑为股票代码为索引，日期为列
        close_prices = historical_prices.pivot(index="code", columns="date", values="close")
        close_prices = close_prices.dropna(axis=0)

        # 计算20日移动平均
        ma20 = close_prices.rolling(window=20, axis=1).mean().iloc[:, -count:]

        # 获取最新一天的收盘价
        last_close_prices = close_prices.iloc[:, -count:]

        # 计算偏离程度（当前收盘价是否大于20日均线）
        bias = last_close_prices > ma20

        # 获取股票所属行业
        industries = self.getStockIndustry(stocks)
        bias["industry_name"] = industries

        # 按行业统计偏离股票的比例
        industry_bias_sum = bias.groupby("industry_name").sum()
        industry_bias_count = bias.groupby("industry_name").count()
        df_ratio = (industry_bias_sum * 100.0 / industry_bias_count).round()

        # 获取偏离比例最高的行业
        top_values = df_ratio.loc[:, yesterday].nlargest(max_industry_cnt)
        top_industries = top_values.index.tolist()

        # 计算全市场宽度的平均偏离比例
        market_width = df_ratio.sum(axis=0).mean()

        log.info(
            [name for name in top_industries],
            "  全市场宽度：",
            market_width
        )

        return top_industries

    def getStockIndustry(self, stocks):
        industry = get_industry(stocks)
        dict = {
            stock: info["sw_l1"]["industry_name"]
            for stock, info in industry.items()
            if "sw_l1" in info
        }
        return pd.Series(dict)

    # 计算市场温度
    def Market_temperature(self, context, market_temperature='warm'):
        log.info(self.name, '--Market_temperature函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        # 获取数据：使用attribute_history函数获取沪深300指数过去220天的收盘价数据。
        index300 = attribute_history('000300.XSHG', 220, '1d', ('close'), df=False)['close']

        # 计算市场高度：通过计算最近5天收盘价的平均值与过去220天收盘价的最小值之差，再除以过去220天收盘价的最大值与最小值之差，得到市场高度（market_height)。
        market_height = (mean(index300[-5:]) - min(index300)) / (max(index300) - min(index300))

        # 判断市场温度：根据市场高度的值，将市场温度分为三种状态：
        # ·如果市场高度小于0.20, 则市场温度为"cold"。
        # ·如果市场高度大于0.90, 则市场温度为"hot"。
        # ·如果过去60天内的最高收盘价与最低收盘价之比大于1.20, 则市场温度为"warm"
        if market_height < 0.20:
            market_temperature = "cold"
        elif market_height > 0.90:
            market_temperature = "hot"
        elif max(index300[-60:]) / min(index300) > 1.20:
            market_temperature = "warm"

        return market_temperature

    # 4-1 打印每日持仓信息,暂无使用
    def print_position_info(self, context):
        log.info(self.name, '--print_position_info函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 打印当天成交记录
        trades = get_trades()
        for _trade in trades.values():
            log.info('成交记录：' + str(_trade))
        # 打印账户信息
        for position in list(context.portfolio.positions.values()):
            securities = position.security
            cost = position.avg_cost
            price = position.price
            ret = 100 * (price / cost - 1)
            value = position.value
            amount = position.total_amount
            log.info('代码:{}'.format(securities))
            log.info('成本价:{}'.format(format(cost, '.2f')))
            log.info('现价:{}'.format(price))
            log.info('收益率:{}%'.format(format(ret, '.2f')))
            log.info('持仓(股):{}'.format(amount))
            log.info('市值:{}'.format(format(value, '.2f')))
            log.info('———————————————————————————————————')
        log.info('———————————————————————————————————————分割线————————————————————————————————————————')

    # 筛选出某一日涨停的股票
    def get_hl_stock(self, context, stock_list, end_date):
        if not stock_list: return []
        h_s = get_price(stock_list, end_date=end_date, frequency='daily', fields=['close', 'high_limit', 'paused'],
                        count=1, panel=False, fill_paused=False, skip_paused=False
                        ).query('close==high_limit and paused==0').groupby('code').size()
        return h_s.index.tolist()

    # 筛选出某一日曾经涨停的股票，含炸板的
    def get_ever_hl_stock(self, context, stock_list, end_date):
        if not stock_list: return []
        h_s = get_price(stock_list, end_date=end_date, frequency='daily', fields=['high', 'high_limit', 'paused'],
                        count=1, panel=False, fill_paused=False, skip_paused=False
                        ).query('high==high_limit and paused==0').groupby('code').size()
        return h_s.index.tolist()

    # 筛选出某一日曾经涨停但未封板的股票
    def get_ever_hl_stock2(self, context, stock_list, end_date):
        if not stock_list: return []
        h_s = get_price(stock_list, end_date=end_date, frequency='daily',
                        fields=['close', 'high', 'high_limit', 'paused'],
                        count=1, panel=False, fill_paused=False, skip_paused=False
                        ).query('close!=high_limit and high==high_limit and paused==0').groupby('code').size()
        return h_s.index.tolist()

    def balance_subportfolios(self, context):
        log.info(self.name, '--balance_subportfolios平衡账户资金函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        length = len(context.portfolio_value_proportion)
        # 计算平衡前仓位比例
        log.info(
            "仓位计划调整的比例："
            + str(
                g.portfolio_value_proportion
            )
            +
            "仓位调整前的比例："
            + str(
                [
                    context.subportfolios[i].total_value / context.portfolio.total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整前的总金额："
            + str(
                [
                    context.subportfolios[i].total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整前的可用金额："
            + str(
                [
                    context.subportfolios[i].available_cash
                    for i in range(length)
                ]
            )
        )
        # 先把所有可用资金打入一号资金仓位
        for i in range(1, length):
            target = context.portfolio_value_proportion[i] * context.portfolio.total_value
            value = context.subportfolios[i].total_value
            if context.subportfolios[i].available_cash > 0 and target < value:
                transfer_cash(
                    from_pindex=i,
                    to_pindex=0,
                    cash=min(value - target, context.subportfolios[i].available_cash),
                )
        # 如果子账户仓位过低，从一号仓位往其中打入资金
        for i in range(1, length):
            target = context.portfolio_value_proportion[i] * context.portfolio.total_value
            value = context.subportfolios[i].total_value
            if target > value and context.subportfolios[0].available_cash > 0:
                transfer_cash(
                    from_pindex=0,
                    to_pindex=i,
                    cash=min(target - value, context.subportfolios[0].available_cash),
                )
        # 计算平衡后仓位比例
        log.info(
            "仓位调整后的比例："
            + str(
                [
                    context.subportfolios[i].total_value / context.portfolio.total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整后的金额："
            + str(
                [
                    context.subportfolios[i].total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整后的可用金额："
            + str(
                [
                    context.subportfolios[i].available_cash
                    for i in range(length)
                ]
            )
        )

    def balance_subportfolios_by_small_to_other(self, context):
        current_month = context.current_dt.month
        if current_month not in (1, 4):
            return
        log.info(self.name, '--balance_subportfolios_by_small_to_other 择时资金转出--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        length = len(context.portfolio_value_proportion)
        # 计算平衡前仓位比例
        log.info(
            "仓位调整前的比例："
            + str(
                [
                    context.subportfolios[i].total_value / context.portfolio.total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整前的总金额："
            + str(
                [
                    context.subportfolios[i].total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整前的可用金额："
            + str(
                [
                    context.subportfolios[i].available_cash
                    for i in range(length)
                ]
            )
        )
        if current_month in (1, 4):
            # 把小市值仓位的资金均分到大市值和ETF
            transfer_dict = {}
            available_cash = context.subportfolios[length - 1].available_cash
            for i in range(0, length - 1):
                log.info('第', length - 1, '个仓位当前可用金额:', available_cash,
                         '，按【', context.portfolio_value_proportion[i], '】比例转到仓位', i)
                value = available_cash * (context.portfolio_value_proportion[i] / 0.5)
                transfer_dict[i] = value
                if value > 0:
                    transfer_cash(
                        from_pindex=length - 1,  ## 小市值
                        to_pindex=i,  ## 大市值 && ETF
                        cash=value
                    )
                    log.info('第', length - 1, '个仓位给第', i, '个仓位转账:', value)
            context.balance_value[current_month] = transfer_dict.copy()
        # 计算平衡后仓位比例
        log.info(
            "仓位调整后的比例："
            + str(
                [
                    context.subportfolios[i].total_value / context.portfolio.total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整后的金额："
            + str(
                [
                    context.subportfolios[i].total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整后的可用金额："
            + str(
                [
                    context.subportfolios[i].available_cash
                    for i in range(length)
                ]
            )
        )

    def balance_subportfolios_by_other_to_small(self, context):
        current_month = context.current_dt.month
        if current_month not in (2, 5):
            return
        log.info(self.name, '--balance_subportfolios_by_other_to_small 择时资金转入--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        length = len(context.portfolio_value_proportion)
        # 计算平衡前仓位比例
        log.info(
            "仓位调整前的比例："
            + str(
                [
                    context.subportfolios[i].total_value / context.portfolio.total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整前的金额："
            + str(
                [
                    context.subportfolios[i].total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整前的可用金额："
            + str(
                [
                    context.subportfolios[i].available_cash
                    for i in range(length)
                ]
            )
        )
        if current_month in (2, 5):
            # 获取上个月份
            if current_month == 2:
                last_month = 1
            else:
                last_month = 4
            # 检查是否有上个月的转账记录
            if last_month in context.balance_value:
                transfer_dict = context.balance_value[last_month]
                # 把小市值仓位的资金归还回来
                for i in transfer_dict:
                    value = transfer_dict[i]
                    if value > 0:
                        transfer_cash(
                            from_pindex=i,
                            to_pindex=length - 1,  ## 小市值
                            cash=value
                        )
                        log.info('第', i, '个仓位给第', length - 1, '个仓位转账:', value)
                # 删除上个月的转账记录
                del context.balance_value[last_month]
            else:
                log.info('没有上个月的转账记录，无需归还')
        # 计算平衡后仓位比例
        log.info(
            "仓位调整后的比例："
            + str(
                [
                    context.subportfolios[i].total_value / context.portfolio.total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整后的金额："
            + str(
                [
                    context.subportfolios[i].total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整后的可用金额："
            + str(
                [
                    context.subportfolios[i].available_cash
                    for i in range(length)
                ]
            )
        )


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


        self.use_wide_time_strategies = self.params['use_wide_time_strategies'] if 'use_wide_time_strategies' in self.params else False  # 是否使用宽度择时
        self.buy_strategy_mode = self.params[
            'buy_strategy_mode'] if 'buy_strategy_mode' in self.params else 'equal'  # 最大持股数
        self.max_hold_count = self.params['max_hold_count'] if 'max_hold_count' in self.params else 1  # 最大持股数
        self.max_select_count = self.params['max_select_count'] if 'max_select_count' in self.params else 5  # 最大输出选股数
        self.hold_limit_days = self.params['hold_limit_days'] if 'hold_limit_days' in self.params else 20  # 计算最近持有列表的天数
        self.use_empty_month = self.params['use_empty_month'] if 'use_empty_month' in self.params else False  # 是否有空仓期
        self.empty_month = self.params['empty_month'] if 'empty_month' in self.params else []  # 空仓月份
        self.use_stoplost = self.params['use_stoplost'] if 'use_stoplost' in self.params else False  # 是否使用止损
        self.empty_month_last_day = self.params[
            'empty_month_last_day'] if 'empty_month_last_day' in self.params else []  # 需要月末清仓的月份
        self.use_empty_month_last_day = self.params[
            'use_empty_month_last_day'] if 'use_empty_month_last_day' in self.params else False  # 是否月末最后一天清仓
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
                  is_filter_lowlimit=True, is_filter_new=True,all_filter=False):
        log.info(self.name, '--stockpool函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        if index is None:
            lists = list(get_all_securities(types=['stock'], date=context.previous_date).index)
        else:
            lists = list(get_index_stocks(index))

        if pool_id == 0:
            pass
        elif pool_id == 1:
            if all_filter:
                lists = self.utilstool.filter_basic_stock(context, lists)
            else:
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
            self.select_list = ['511880.XSHG']
            return
        # 止损期控制
        if self.use_stoplost and self.stoplost_date is not None:
            self.select_list = ['511880.XSHG']
            return
        self.select_list = []

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
    def close_for_empty_month(self, context, exempt_stocks=None):
        if exempt_stocks is None:
            exempt_stocks = ['511880.XSHG']

        log.info(self.name, f'--close_for_empty_month函数：在空仓期保留{exempt_stocks}，卖出其他股票--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        subportfolio = context.subportfolios[self.subportfolio_index]
        if self.use_empty_month and context.current_dt.month in self.empty_month and len(
                subportfolio.long_positions) > 0:
            # 获取当前持有的所有股票
            positions = list(subportfolio.long_positions)
            # 排除exempt_stocks中的股票
            stocks_to_sell = [stock for stock in positions if stock not in exempt_stocks]
            if stocks_to_sell:
                self.sell(context, stocks_to_sell)
                log.info(self.name, f'--空仓期卖出股票：{stocks_to_sell}，保留{exempt_stocks}--',
                         str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
            else:
                log.info(self.name, f'--空仓期没有需要卖出的股票，保留{exempt_stocks}--',
                         str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

    # 每月最后一天，清仓等账户均衡
    def close_for_month_last_day(self, context):
        log.info(self.name, '--close_for_month_last_day函数，每月最后一天，清仓等账户均衡--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        subportfolio = context.subportfolios[self.subportfolio_index]
        if self.use_empty_month_last_day and context.current_dt.month in (self.empty_month_last_day) and len(
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
                # 获取持仓股票的当前最新价
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
                if (context.current_dt + dt.timedelta(
                        days=-self.stoplost_silent_days)).date() >= self.stoplost_date:
                    self.stoplost_date = None
                    log.info(self.name + ': ' + '退出止损')

    # 止损时清仓
    def close_for_stoplost(self, context, exempt_stocks=None):
        if exempt_stocks is None:
            exempt_stocks = ['511880.XSHG']

        log.info(self.name, f'--close_for_stoplost函数：在止损期保留{exempt_stocks}，卖出其他股票--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        subportfolio = context.subportfolios[self.subportfolio_index]
        if self.use_stoplost and self.stoplost_date is not None and len(subportfolio.long_positions) > 0:
            # 获取当前持有的所有股票
            positions = list(subportfolio.long_positions)
            # 排除exempt_stocks中的股票
            stocks_to_sell = [stock for stock in positions if stock not in exempt_stocks]
            if stocks_to_sell:
                self.sell(context, stocks_to_sell)
                log.info(self.name, f'--止损期卖出股票：{stocks_to_sell}，保留{exempt_stocks}--',
                         str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
            else:
                log.info(self.name, f'--止损期没有需要卖出的股票，保留{exempt_stocks}--',
                         str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

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
    def adjustwithnoRM(self, context, only_buy=False, only_sell=False, together=True, is_single_buy=False,
                       exempt_stocks=None):
        log.info(self.name, '--adjustwithnoRM调仓函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        if exempt_stocks is None:
            exempt_stocks = ['511880.XSHG']

        # 空仓期或者止损期不再进行调仓
        if self.use_empty_month and context.current_dt.month in (self.empty_month):
            log.info('adjustwithnoRM调仓函数不再执行，因为当前月份是空仓期，空仓期月份为：', self.empty_month)
            self.buy(context, exempt_stocks, is_single_buy)
            return
        # 止损期控制
        if self.use_stoplost and self.stoplost_date is not None:
            log.info('adjustwithnoRM调仓函数不再执行，因为当前时刻还处于止损期，止损期从:', self.stoplost_date, '开始')
            self.buy(context, exempt_stocks, is_single_buy)
            return

        # 先卖后买
        hold_list = list(context.subportfolios[self.subportfolio_index].long_positions)
        # 售卖列表：不在select_list前max_hold_count中的股票都要被卖掉
        sell_stocks = []
        # 实时过滤部分股票，否则也买不了，放出去也没有意义
        target_list = self.utilstool.filter_highlimit_stock(context, self.select_list)
        target_list = self.utilstool.filter_paused_stock(context, target_list)
        # target_list = self.utilstool.filter_lowlimit_stock(context, target_list)

        log.info(self.name, '--过滤部分股票后的选股列表:', target_list)
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

    # 调仓+均衡资产
    def adjustwithnoRMBalance(self, context, only_buy=False, only_sell=False, together=True, is_single_buy=False,
                              exempt_stocks=None):
        log.info(self.name, '--adjustwithnoRMBalance调仓函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        if exempt_stocks is None:
            exempt_stocks = ['511880.XSHG']

        # 空仓期或者止损期不再进行调仓
        if self.use_empty_month and context.current_dt.month in (self.empty_month):
            log.info('adjustwithnoRM调仓函数不再执行，因为当前月份是空仓期，空仓期月份为：', self.empty_month)
            self.buy(context, exempt_stocks, is_single_buy)
            return
        # 止损期控制
        if self.use_stoplost and self.stoplost_date is not None:
            log.info('adjustwithnoRM调仓函数不再执行，因为当前时刻还处于止损期，止损期从:', self.stoplost_date, '开始')
            self.buy(context, exempt_stocks, is_single_buy)
            return

        # 先卖后买
        hold_list = list(context.subportfolios[self.subportfolio_index].long_positions)
        # 售卖列表：不在select_list前max_hold_count中的股票都要被卖掉
        sell_stocks = []
        # 实时过滤部分股票，否则也买不了，放出去也没有意义
        target_list = self.utilstool.filter_highlimit_stock(context, self.select_list)
        target_list = self.utilstool.filter_paused_stock(context, target_list)
        # target_list = self.utilstool.filter_lowlimit_stock(context, target_list)

        log.info(self.name, '--过滤部分股票后的选股列表:', target_list)
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
            self.balance_subportfolios(context)
            self.buy(context, target_list, is_single_buy)
            return

    # 平衡账户间资金
    def balance_subportfolios(self, context):
        log.info(f"{self.name}"
                 f"--仓位计划调整的比例:{g.portfolio_value_proportion[self.subportfolio_index]}"
                 f"--仓位调整前的总金额:{context.subportfolios[self.subportfolio_index].total_value}"
                 f"--仓位调整前的可用金额:{context.subportfolios[self.subportfolio_index].available_cash}"
                 f"--仓位调整前的可取金额:{context.subportfolios[self.subportfolio_index].transferable_cash}"
                 f"--仓位调整前的比例:{context.subportfolios[self.subportfolio_index].total_value / context.portfolio.total_value}"
                 )
        target = (
                g.portfolio_value_proportion[self.subportfolio_index]
                * context.portfolio.total_value
        )
        value = context.subportfolios[self.subportfolio_index].total_value
        # 仓位比例过高调出资金
        cash = context.subportfolios[self.subportfolio_index].transferable_cash  # 当前账户可取资金
        if cash > 0 and target < value:
            amount = min(value - target, cash)
            transfer_cash(
                from_pindex=self.subportfolio_index,
                to_pindex=0,
                cash=amount,
            )
            log.info('第', self.subportfolio_index, '个仓位调整了【', amount, '】元到仓位：0')
            self.get_net_values(context, -amount)

        # 仓位比例过低调入资金
        cash = context.subportfolios[0].transferable_cash  # 0号账户可取资金
        if target > value and cash > 0:
            amount = min(target - value, cash)
            transfer_cash(
                from_pindex=0,
                to_pindex=self.subportfolio_index,
                cash=amount,
            )
            log.info('第0个仓位调整了【', amount, '】元到仓位：', self.subportfolio_index)
            self.get_net_values(context, amount)

    # 计算策略复权后净值
    def get_net_values(self, context, amount):
        df = g.strategys_values
        if df.empty:
            return
        column_index = self.subportfolio_index - 1
        # 获取最后一天的索引

        last_day_index = len(df) - 1
        # 调整最后一天的净值
        df.iloc[last_day_index, column_index] -= amount

        # 计算调整后的最后一天的净值
        final_value = df.iloc[last_day_index, column_index]

        # 调整前面的净值，保持收益率不变
        for i in range(last_day_index - 1, -1, -1):
            df.iloc[i, column_index] = final_value * (
                    df.iloc[i, column_index] / df.iloc[last_day_index, column_index]
            )

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
            pass

        current_holdings = subportfolio.long_positions
        available_cash = subportfolio.available_cash
        max_hold_count = self.max_hold_count
        current_holding_count = len(current_holdings)

        # 分离buy_stocks为已持仓和未持仓两部分
        held_stocks = [stock for stock in buy_stocks if stock in current_holdings]
        new_stocks = [stock for stock in buy_stocks if stock not in current_holdings]

        # 计算可以买入的未持仓股票数量
        total_new = min(max_hold_count - current_holding_count, len(new_stocks))
        total_held = len(held_stocks)
        log.info(self.buy_strategy_mode, '策略详情:目标股票列表--', buy_stocks,
                 '--最大持仓股票数--', max_hold_count,
                 '--当前持仓股票数--', current_holding_count,
                 '--当前持仓股票明细--', current_holdings,
                 '--目标股票中未持仓股票列表--', new_stocks,
                 '--目标股票中已持仓股票列表--', held_stocks
                 )

        log.info(self.buy_strategy_mode, '策略详情:当前持仓--', current_holdings, '--已持仓股票列表--', held_stocks,
                 '--未持仓股票列表--', new_stocks)

        if self.buy_strategy_mode == 'equal':
            # Strategy 1: Buy new and held stocks equally
            # 计算总的购买金额
            total_value = available_cash
            if (total_new + total_held) <= 0 or total_value <= 0:
                log.info('没有可购买的股票。')
                return

            stock_value = total_value / (total_new + total_held)
            log.debug('equal买入策略：计算总的购买金额：', total_value)
            log.debug('equal买入策略：每只股票的购买金额比例：', stock_value)
            log.debug('equal买入策略：计算可以买入的未持仓股票数量：', total_new, '--待买入列表:', new_stocks)
            log.debug('equal买入策略：计算可以买入的已持仓股票数量：', total_held, '--已持仓列表:', held_stocks)

            # 加仓已持有的股票
            if total_held > 0:
                for stock in held_stocks:
                    if available_cash <= 0:
                        break
                    value = min(stock_value, available_cash)
                    if self.utilstool.open_position(context, stock, value, False):
                        available_cash -= value
                        log.info(f'加仓已持有股票 {stock}，金额: {value}')
                    else:
                        log.warning(f'加仓已持有股票 {stock} 失败，跳过。')

            # 购买新股票
            if total_new > 0:
                for stock in new_stocks:
                    if available_cash <= 0:
                        break
                    value = min(stock_value, available_cash)
                    if self.utilstool.open_position(context, stock, value, False):
                        available_cash -= value
                        log.info(f'买入新股票 {stock}，金额: {value}')
                    else:
                        log.warning(f'买入新股票 {stock} 失败，跳过。')


        elif self.buy_strategy_mode == 'priority':
            # Strategy 2: Prioritize new stocks, then held stocks
            if total_new > 0:
                stock_value = available_cash / total_new
                log.debug('priority买入策略：计算总的购买金额：', available_cash)
                log.debug('priority买入策略：每只股票的购买金额比例：', stock_value)
                log.debug('priority买入策略：计算可以买入的未持仓股票数量：', total_new, '--待买入列表:', new_stocks)
                for stock in new_stocks:
                    if available_cash <= 0:
                        break
                    value = min(stock_value, available_cash)
                    if self.utilstool.open_position(context, stock, value, False):
                        available_cash -= value
                        log.info(f'买入新股票 {stock}，金额: {value}')
                    else:
                        log.warning(f'买入新股票 {stock} 失败，跳过。')

            if total_held > 0:
                stock_value = available_cash / total_held
                log.debug('priority买入策略：计算总的购买金额：', available_cash)
                log.debug('priority买入策略：每只股票的购买金额比例：', stock_value)
                log.debug('priority买入策略：计算可以买入的已持仓股票数量：', total_held, '--待买入列表:', held_stocks)
                for stock in held_stocks:
                    if available_cash <= 0:
                        break
                    value = min(stock_value, available_cash)
                    if self.utilstool.open_position(context, stock, value, False):
                        available_cash -= value
                        log.info(f'加仓已持有股票 {stock}，金额: {value}')
                    else:
                        log.warning(f'加仓已持有股票 {stock} 失败，跳过。')

        else:
            log.warning('无效的策略模式。')
            return

    # 卖出多只股票
    def sell(self, context, sell_stocks):

        log.info(self.name, '--sell函数--要卖出的股票列表--', sell_stocks,
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

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

# 白马股攻防转换策略（BMZH策略）
class BMZH_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)
        self.market_temperature = "warm"

    def select(self, context):
        log.info(self.name, '--select函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        self.market_temperature = self.utilstool.Market_temperature(context, self.market_temperature)
        # 根据市场温度设置选股条件，选出股票
        self.select_list = self.__get_rank(context)[:self.max_select_count]
        log.info('当前市场温度:', self.market_temperature, '当前选股:', self.select_list)
        # 编写操作计划
        self.print_trade_plan(context, self.select_list)

    def __get_rank(self, context):
        log.info(self.name, '--get_rank函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        initial_list = super().stockpool(context, 1, "000300.XSHG")
        # log.info(initial_list)
        # 2.根据市场温度进行选股
        # ·如果市场温度为"cold",则筛选条件包括：
        #   市净率（PB ratio)大于0且小于1
        #   经营活动现金流入小计大于0
        #   扣除非经常损益后的净利润大于2.5亿
        #   营业收入大于10亿
        #   净利润大于2.5亿
        #   经营活动现金流入小计与扣除非经常损益后的净利润之比大于2.0
        #   净资产收益率（ROA)大于1.5
        #   净利润同比增长率大于-15%
        #   并且股票代码在初始列表中。
        # 查询结果按照ROA与市净率的比值降序排列
        # 并限制最多返回 self.max_select_count+1只股票。
        if self.market_temperature == "cold":
            q = query(
                valuation.code,
            ).filter(
                valuation.pb_ratio > 0,
                valuation.pb_ratio < 1,
                cash_flow.subtotal_operate_cash_inflow > 0,  # 经营活动现金流入小计
                indicator.adjusted_profit > 2.5e8,  # 扣除非经常损益后的净利润(元)                          #>=2.5亿
                income.operating_revenue > 10e8,  # 营业收入(元)                                          #>=10亿
                income.net_profit > 2.5e8,  # 净利润(元)                                            #>=2.5亿
                cash_flow.subtotal_operate_cash_inflow / indicator.adjusted_profit > 2.0,  # 经营活动现金流入小计/扣除非经常损益后的净利润(元)
                indicator.inc_return > 1.5,  # 净资产收益率(扣除非经常损益)(%)
                indicator.inc_net_profit_year_on_year > -15,  # 净利润同比增长率(%)
                valuation.code.in_(initial_list)
            ).order_by(
                (indicator.roa / valuation.pb_ratio).desc()
            ).limit(self.max_select_count + 1)

        # 如果市场温度为"warm"
        # 则筛选条件与"cold"类似，但经营活动现金流入小计与扣除非经常损益后的净利润之比大于1.0
        #   净资产收益率大于2.0
        #   净利润同比增长率大于0%。

        elif self.market_temperature == "warm":
            q = query(
                valuation.code,
            ).filter(
                valuation.pb_ratio > 0,
                valuation.pb_ratio < 1,
                cash_flow.subtotal_operate_cash_inflow > 0,
                indicator.adjusted_profit > 2.5e8,  # 扣除非经常损益后的净利润(元)                          #>=2.5亿
                income.operating_revenue > 10e8,  # 营业收入(元)                                          #>=10亿
                income.net_profit > 2.5e8,  # 净利润(元)                                            #>=2.5亿
                cash_flow.subtotal_operate_cash_inflow / indicator.adjusted_profit > 1.0,
                indicator.inc_return > 2.0,
                indicator.inc_net_profit_year_on_year > 0,
                valuation.code.in_(initial_list)
            ).order_by(
                (indicator.roa / valuation.pb_ratio).desc()
            ).limit(self.max_select_count + 1)
        # 如果市场温度为hot, 则筛选条件包括：
        #   市净率大于3,
        #   经营活动现金流入小计大于0
        #   扣除非经常损益后的净利润大于2.5亿
        #   营业收入大于10亿
        #   净利润大于2.5亿
        #   经营活动现金流入小计与除非经常损益后的净利润之比大于0.5
        #   净资产收益率大于3.0
        #   净利润同比增长率大于20%
        #   并且股票代码在初始列表中
        # 查询结果按照ROA降序排列
        # 并限制最多返回 self.max_select_count+1只股票。
        elif self.market_temperature == "hot":
            q = query(
                valuation.code,
            ).filter(
                valuation.pb_ratio > 3,
                cash_flow.subtotal_operate_cash_inflow > 0,
                indicator.adjusted_profit > 2.5e8,  # 扣除非经常损益后的净利润(元)                          #>=2.5亿
                income.operating_revenue > 10e8,  # 营业收入(元)                                          #>=10亿
                income.net_profit > 2.5e8,  # 净利润(元)                                            #>=2.5亿
                cash_flow.subtotal_operate_cash_inflow / indicator.adjusted_profit > 0.5,
                indicator.inc_return > 3.0,
                indicator.inc_net_profit_year_on_year > 20,
                valuation.code.in_(initial_list)
            ).order_by(
                indicator.roa.desc()
            ).limit(self.max_select_count + 1)

        # 得到选股列表
        # 3.执行查询并获取选股列表：使用 get_fundamentals 函数执行查询，并将查询结果转换为股票代码列表，然后返回这个列表。
        check_out_lists = list(get_fundamentals(q).code)
        return check_out_lists


# 外盘ETF轮动策略
class WPETF_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)
        self.foreign_ETF = [
            '518880.XSHG',  # 黄金
            '513030.XSHG',  # 德国
            '513100.XSHG',  # 纳指
            '164824.XSHE',  # 印度
            '159866.XSHE',  # 日本

            '513500.XSHG',  # 标普500
            '159915.XSHE'  # 创业板100
            # '161716.XSHE',#招商双债
        ]
        self.deltaday = 20  # 上市天数
        self.days = 14  # 计算ATR的序列长度

    def select(self, context):
        log.info(self.name, '--Select函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        self.select_list = self.__get_rank(context)[:self.max_select_count]
        self.print_trade_plan(context, self.select_list)

    def __get_rank(self, context):
        log.info(self.name, '--get_rank函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        etf_pool = self.fun_delNewShare(context, self.foreign_ETF, self.deltaday)
        score_list = []
        if len(etf_pool) == 0:
            # 如果没有合适的 ETF 就买入国债
            etf_pool = self.fun_delNewShare(context, ['511010.XSHG', '511880.XSHG'], self.deltaday)
            if len(etf_pool) == 0:  # 2013年前的测试会出现这种情况
                log.info('ETF_pool 为空！')
            final_list = etf_pool
            return final_list
        for etf in etf_pool:
            try:
                # 计算ATR
                atr = self.getATR(context, etf, period=self.days)
                score_list.append(atr)
            except ValueError as e:
                log.error(e)
                score_list.append(np.nan)
        df = pd.DataFrame(index=etf_pool, data={'ATR': score_list})
        # 删除包含 NaN 值的行
        df = df.dropna()
        df = df.sort_values(by='ATR', ascending=True)
        final_list = list(df.index)
        log.info("——————————————————————————————————")
        for i, etf in enumerate(df.index):
            name = get_security_info(etf).display_name
            log.info("编号:{}. 股票:{}，ATR:{}".format(i + 1, name, df.loc[etf, 'ATR']))
        log.info("——————————————————————————————————")
        return final_list

    # 2 全球ETF 平均真实波幅（ATR）
    def getATR(self, context, stock, period=14):
        log.info(self.name, '--getATR函数--计算', stock, '的 ATR信息--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 获取历史数据
        hData = attribute_history(stock, period + 1, unit='1d',
                                  fields=('close', 'volume', 'open', 'high', 'low'),
                                  skip_paused=True, df=False)
        high = hData['high']
        low = hData['low']
        close = hData['close']
        # 检查并处理 NaN 值
        if any(np.isnan(high)) or any(np.isnan(low)) or any(np.isnan(close)):
            raise ValueError(f"{stock}的历史数据包含NaN(非数字)值。")
        # 计算ATR
        realATR = tl.ATR(high, low, close, timeperiod=period)
        realATR = realATR / close.mean()
        return realATR[-1]

    #############################外盘ETF策略增加通用函数###########################
    # 删除上市少于deltaday天的股票
    def fun_delNewShare(self, context, equity, deltaday):
        log.info(self.name, '--fun_delNewShare函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        deltaDate = context.current_dt.date() - dt.timedelta(deltaday)
        tmpList = []
        for stock in equity:
            stock_info = get_security_info(stock)
            if stock_info is not None and stock_info.start_date < deltaDate:
                tmpList.append(stock)
        return tmpList


class XSZ_GJT_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)
        self.highest = 50

    def select(self, context):
        log.info(self.name, '--select选股函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 空仓期控制和止损期都不在选择股票
        if self.use_empty_month and context.current_dt.month in (self.empty_month):
            log.info('Select选股函数不再执行，因为当前月份是空仓期，空仓期月份为：', self.empty_month)
            return
        if self.stoplost_date is not None:
            log.info('Select选股函数不再执行，因为当前时刻还处于止损期，止损期从:', self.stoplost_date, '开始')
            return

        self.select_list = self.__get_rank(context)[:self.max_select_count]
        log.info(self.name, '的选股列表:', self.select_list)
        self.print_trade_plan(context, self.select_list)

    def __get_rank(self, context):
        log.info(self.name, '--get_rank函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 获得初始列表
        initial_list = self.stockpool(context, 1, '399101.XSHE')
        # 过滤120天内即将大幅解禁
        initial_list = self.utilstool.filter_locked_shares(context, initial_list, 120)

        final_list_1 = []
        # 市值5-30亿，并且在列表中，按市值从小到大到排序
        q = (query(valuation.code, valuation.market_cap)
             .filter(valuation.code.in_(initial_list), valuation.market_cap.between(5, 30))
             .order_by(valuation.market_cap.asc()))

        # 获取财务数据
        df_1 = get_fundamentals(q)[:50]
        # log.info(self.name, '过滤停盘/涨停/跌停之后，--前50股票的财务数据:', df_fun)
        final_list_1 = list(df_1.code)

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
            valuation.code.in_(initial_list),
            valuation.market_cap.between(5, 30),
            income.np_parent_company_owners > 0,
            income.net_profit > 0,
            income.operating_revenue > 1e8
        ).order_by(valuation.market_cap.asc()).limit(50)

        df_2 = get_fundamentals(q)

        final_list_2 = list(df_2.code)
        last_prices = history(1, unit='1d', field='close', security_list=final_list_2)
        # 过滤价格低于最高价50元/股的股票  ｜  再持仓列表中的股票
        final_list_2 = [stock for stock in final_list_2 if
                        stock in self.hold_list or last_prices[stock][-1] <= self.highest]

        # 合并两个股票列表并去重
        target_list = list(dict.fromkeys(final_list_1 + final_list_2))
        # 取前 self.max_select_count * 3 只股票
        target_list = target_list[:self.max_select_count * 3]
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

