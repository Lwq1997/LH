'''
多策略分子账户并行
用到的策略：
蚂蚁量化,东哥：白马股攻防转换策略（BMZH策略）
linlin2018，ZLH：低波全天候策略（外盘ETF策略）
@荒唐的方糖大佬:国九条小市值（XSZGJT）（还可以改进）
'''

# 导入函数库
#-*- coding: utf-8 -*-
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
from BMZH_Strategy import BMZH_Strategy
from WPETF_Strategy import WPETF_Strategy
from XSZ_GJT_Strategy import XSZ_GJT_Strategy


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
    set_slippage(PriceRelatedSlippage(0.01), type='stock')

    # 临时变量

    # 持久变量
    g.strategys = {}
    # 子账户 分仓
    g.portfolio_value_proportion = [0.35, 0.15, 0.5]

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
        'max_hold_count': 2,  # 最大持股数
        'max_select_count': 4,  # 最大输出选股数
    }
    # 白马策略，第一个仓
    bmzh_strategy = BMZH_Strategy(context, subportfolio_index=0, name='白马股攻防转换策略', params=params)
    g.strategys[bmzh_strategy.name] = bmzh_strategy

    params = {
        'max_hold_count': 2,  # 最大持股数
        'max_select_count': 4,  # 最大输出选股数
    }
    # ETF 策略，第二个仓
    wpetf_strategy = WPETF_Strategy(context, subportfolio_index=1, name='外盘ETF轮动策略', params=params)
    g.strategys[wpetf_strategy.name] = wpetf_strategy

    params = {
        'max_hold_count': 3,  # 最大持股数
        'max_select_count': 6,  # 最大输出选股数
        'use_empty_month': True,  # 是否在指定月份空仓
        'empty_month': [1, 4]  # 指定空仓的月份列表
        # 'use_stoplost': True,  # 是否使用止损
    }
    # 小世值，第三个仓
    xszgjt_strategy = XSZ_GJT_Strategy(context, subportfolio_index=2, name='国九条小市值策略', params=params)
    g.strategys[xszgjt_strategy.name] = xszgjt_strategy

    # 执行计划
    # 选股函数--Select：白马和 ETF 分开使用
    # 执行函数--adjust：白马和 ETF 轮动共用一个
    # # 白马，按月运行 TODO
    if g.portfolio_value_proportion[0] > 0:
        run_monthly(bmzh_select, 1, time='7:40')  # 阅读完成，测试完成
        run_monthly(bmzh_adjust, 1, time='09:30')  # 阅读完成，测试完成
        run_daily(bmzh_after_market_close, 'after_close')
    #
    # # ETF轮动，按天运行
    if g.portfolio_value_proportion[1] > 0:
        run_daily(wpetf_select, time='7:42')  # 阅读完成，测试完成
        run_daily(wpetf_adjust, time='09:30')  # 阅读完成，测试完成
        run_daily(wpetf_after_market_close, 'after_close')

    # # 小市值，按天/周运行
    if g.portfolio_value_proportion[2] > 0:
        run_daily(xszgjt_day_prepare, time='7:33')
        run_weekly(xszgjt_select, 1, time='7:43')
        run_daily(xszgjt_open_market, time='9:30')
        run_weekly(xszgjt_adjust, 1, time='9:30')
        # run_daily(xszgjt_sell_when_highlimit_open, time='11:27')
        run_daily(xszgjt_sell_when_highlimit_open, time='14:00')
        run_daily(xszgjt_sell_when_highlimit_open, time='14:50')
        run_daily(xszgjt_append_buy_stock, time='14:51')
        run_daily(xszgjt_after_market_close, 'after_close')
        # run_daily(xszgjt_print_position_info, time='15:10')


# # 每个交易日结束运行
# def after_trading_end(context):
#     log.warn('##############################################################')
#     # 得到当天所有成交记录
#     trades = get_trades()
#     for _trade in trades.values():
#         log.warn('成交记录：' + str(_trade))
#     now = str(context.current_dt.date()) + ' ' + str(context.current_dt.time())
#     log.warn('--after_trading_end函数--', now)
#
#     # print(context.subportfolios_name_map)
#     for key, value in context.subportfolios_name_map.items():
#         log.warn('after_trading_end函数----', now, ':账户,', value, '的余额:',
#                  context.subportfolios[key].available_cash)
#         log.warn('after_trading_end函数----', now, ':账户,', value, '的当前持仓:',
#                  context.subportfolios[key].long_positions)
#         log.warn('after_trading_end函数----', now, ':账户,', value, '的账户总资产:',
#                  context.subportfolios[key].total_value)
#
#     log.warn('##############################################################')


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
