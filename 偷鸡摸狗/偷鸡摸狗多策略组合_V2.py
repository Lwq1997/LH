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
from JSG_Strategy import JSG_Strategy
from All_Day_Strategy import All_Day_Strategy
from Rotation_ETF_Strategy import Rotation_ETF_Strategy


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

    # 固定滑点设置股票0.01，基金0.001(即交易对手方一档价)
    set_slippage(FixedSlippage(0.02), type="stock")
    set_slippage(FixedSlippage(0.002), type="fund")
    # 设置股票交易印花税千一，佣金万三
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0.001,
            open_commission=0.0003,
            close_commission=0.0003,
            close_today_commission=0,
            min_commission=5,
        ),
        type="stock",
    )
    # 设置货币ETF交易佣金0
    set_order_cost(
        OrderCost(
            open_tax=0,
            close_tax=0,
            open_commission=0,
            close_commission=0,
            close_today_commission=0,
            min_commission=0,
        ),
        type="mmf",
    )

    # 临时变量

    # 持久变量
    g.strategys = {}
    # 子账户 分仓
    g.portfolio_value_proportion = [0.5, 0, 0.5]

    # 创建策略实例
    # 初始化策略子账户 subportfolios
    set_subportfolios([
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[0], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[1], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[2], 'stock'),
    ])

    # 是否发送微信消息，回测环境不发送，模拟环境发送
    context.is_send_wx_message = 0

    params = {
        'max_select_count': 6,
        'buy_strategy_mode': 'priority',
        'max_hold_count': 6,
        'use_empty_month': True,  # 是否在指定月份空仓
        'empty_month': [1, 4],  # 指定空仓的月份列表
    }
    jsg_strategy = JSG_Strategy(context, subportfolio_index=0, name='搅屎棍策略', params=params)
    g.strategys[jsg_strategy.name] = jsg_strategy

    params = {
    }
    all_day_strategy = All_Day_Strategy(context, subportfolio_index=1, name='全天候策略', params=params)
    g.strategys[all_day_strategy.name] = all_day_strategy

    params = {
        'max_hold_count': 1
    }
    rotation_etf_strategy = Rotation_ETF_Strategy(context, subportfolio_index=2, name='核心资产轮动策略', params=params)
    g.strategys[rotation_etf_strategy.name] = rotation_etf_strategy


# 模拟盘在每天的交易时间结束后会休眠，第二天开盘时会恢复，如果在恢复时发现代码已经发生了修改，则会在恢复时执行这个函数。 具体的使用场景：可以利用这个函数修改一些模拟盘的数据。
def after_code_changed(context):  # 输出运行时间
    log.info('函数运行时间(after_code_changed)：' + str(context.current_dt.time()))

    # 是否发送微信消息，回测环境不发送，模拟环境发送
    context.is_send_wx_message = 0

    unschedule_all()  # 取消所有定时运行

    # 子策略执行计划
    if g.portfolio_value_proportion[0] > 0:
        run_daily(jsg_prepare, "7:00")
        run_weekly(jsg_select, 1, "7:30")
        run_weekly(jsg_open_market, 1, "9:30")
        run_weekly(jsg_adjust, 1, "9:31")
        run_daily(jsg_check, "14:50")

    if g.portfolio_value_proportion[1] > 0:
        run_monthly(all_day_adjust, 1, "9:40")

    if g.portfolio_value_proportion[2] > 0:
        run_daily(rotation_etf_select, "7:30")
        run_daily(rotation_etf_adjust, "9:32")

    run_daily(after_market_close, 'after_close')


def jsg_prepare(context):
    g.strategys["搅屎棍策略"].day_prepare(context)


def jsg_select(context):
    g.strategys["搅屎棍策略"].select(context)


def jsg_adjust(context):
    g.strategys["搅屎棍策略"].adjustwithnoRM(context)


def jsg_check(context):
    g.strategys["搅屎棍策略"].sell_when_highlimit_open(context)


def jsg_open_market(context):
    g.strategys['搅屎棍策略'].close_for_empty_month(context)
    g.strategys['搅屎棍策略'].close_for_stoplost(context)


def all_day_adjust(context):
    g.strategys["全天候策略"].adjust(context)


def rotation_etf_select(context):
    g.strategys["核心资产轮动策略"].select(context)


def rotation_etf_adjust(context):
    g.strategys["核心资产轮动策略"].adjustwithnoRM(context)


def after_market_close(context):
    g.strategys['搅屎棍策略'].after_market_close(context)
    g.strategys['全天候策略'].after_market_close(context)
    g.strategys['核心资产轮动策略'].after_market_close(context)
