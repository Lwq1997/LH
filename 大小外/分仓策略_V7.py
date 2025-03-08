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
from PJ_Strategy2 import PJ_Strategy2
from WP_Strategy import WP_Strategy
from All_Day2_Strategy import All_Day2_Strategy


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
    g.global_sold_stock_record = {}  # 全局卖出记录
    g.strategys = {}
    # 子账户 分仓
    g.portfolio_value_proportion = [0, 0.1, 0.8, 0.1]

    # 创建策略实例
    # 初始化策略子账户 subportfolios
    set_subportfolios([
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[0], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[1], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[2], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[3], 'stock'),
    ])

    params = {
        'max_hold_count': 1,  # 最大持股数
        'max_industry_cnt' : 1, #最大行业数
        'max_select_count': 20,  # 最大输出选股数
    }
    pj_strategy = PJ_Strategy2(context, subportfolio_index=1, name='破净策略', params=params)
    g.strategys[pj_strategy.name] = pj_strategy

    params = {
        'max_hold_count': 6,  # 最大持股数
        'max_select_count': 30,  # 最大输出选股数
        'max_industry_cnt' : 1, #最大行业数
        'use_empty_month': True,  # 是否在指定月份空仓
        'empty_month': [1, 4]  # 指定空仓的月份列表
    }
    wp_strategy = WP_Strategy(context, subportfolio_index=2, name='微盘策略', params=params)
    g.strategys[wp_strategy.name] = wp_strategy

    params = {
    }
    all_day_strategy = All_Day2_Strategy(context, subportfolio_index=3, name='全天候策略', params=params)
    g.strategys[all_day_strategy.name] = all_day_strategy


# 模拟盘在每天的交易时间结束后会休眠，第二天开盘时会恢复，如果在恢复时发现代码已经发生了修改，则会在恢复时执行这个函数。 具体的使用场景：可以利用这个函数修改一些模拟盘的数据。
def after_code_changed(context):  # 输出运行时间
    log.info('函数运行时间(after_code_changed)：' + str(context.current_dt.time()))

    # 是否发送微信消息，回测环境不发送，模拟环境发送
    context.is_send_wx_message = 0

    unschedule_all()  # 取消所有定时运行

    # 设置调仓
    run_monthly(balance_subportfolios, 1, "9:02")  # 资金平衡

    # 破净策略调仓设置
    if g.portfolio_value_proportion[1] > 0:
        run_daily(prepare_pj_strategy, "9:03")
        run_monthly(select_pj_strategy, 1, "9:40")  # 阅读完成，测试完成
        run_monthly(adjust_pj_strategy, 1, "9:40")
        run_daily(pj_sell_when_highlimit_open, time='11:20')
        run_daily(pj_sell_when_highlimit_open, time='14:50')

    # 微盘策略调仓设置
    if g.portfolio_value_proportion[2] > 0:
        run_daily(prepare_wp_strategy, "9:03")
        run_daily(wp_open_market, "9:30")
        run_weekly(select_wp_strategy, 1, "11:00")  # 阅读完成，测试完成
        run_weekly(adjust_wp_strategy, 1, "11:00")
        run_daily(wp_sell_when_highlimit_open, time='11:20')
        run_daily(wp_sell_when_highlimit_open, time='14:50')

    # 全天策略调仓设置
    if g.portfolio_value_proportion[3] > 0:
        run_monthly(adjust_qt_strategy, 1, "10:00")

    # 核心策略调仓设置
    # if g.portfolio_value_proportion[4] > 0:
    #     run_daily(adjust_hx_strategy, "10:05")


# 资金平衡函数==========================================================
def balance_subportfolios(context):

    # g.strategys["破净策略"].balance_subportfolios(context)
    # g.strategys["微盘策略"].balance_subportfolios(context)
    # g.strategys["全天候策略"].balance_subportfolios(context)

    for i in range(1, len(g.portfolio_value_proportion)):
        target = g.portfolio_value_proportion[i] * context.portfolio.total_value
        value = context.subportfolios[i].total_value
        deviation = abs((value - target) / target) if target != 0 else 0
        if deviation > 0.3:
            if context.subportfolios[i].available_cash > 0 and target < value:
                log.info('第', i, '个仓位调整了【', min(value - target, context.subportfolios[i].available_cash), '】元到仓位：0')
                transfer_cash(from_pindex=i, to_pindex=0,
                              cash=min(value - target, context.subportfolios[i].available_cash))
            if target > value and context.subportfolios[0].available_cash > 0:
                log.info('第0个仓位调整了【', min(target - value, context.subportfolios[0].available_cash), '】元到仓位：',i)
                transfer_cash(from_pindex=0, to_pindex=i,
                              cash=min(target - value, context.subportfolios[0].available_cash))


# 破净策略
def prepare_pj_strategy(context):
    g.strategys["破净策略"].day_prepare(context)


def select_pj_strategy(context):
    g.strategys["破净策略"].select(context)


def adjust_pj_strategy(context):
    g.strategys["破净策略"].adjustwithnoRM(context)


def pj_sell_when_highlimit_open(context):
    g.strategys['破净策略'].sell_when_highlimit_open(context)
    if g.strategys['破净策略'].is_stoplost_or_highlimit:
        g.strategys['破净策略'].select(context)
        g.strategys['破净策略'].adjustwithnoRM(context)
        g.strategys['破净策略'].is_stoplost_or_highlimit = False


# 微盘策略
def prepare_wp_strategy(context):
    g.strategys["微盘策略"].day_prepare(context)


def wp_open_market(context):
    g.strategys['微盘策略'].close_for_empty_month(context)
    g.strategys['微盘策略'].close_for_stoplost(context)


def select_wp_strategy(context):
    g.strategys["微盘策略"].select(context)


def adjust_wp_strategy(context):
    g.strategys["微盘策略"].adjustwithnoRM(context)


def wp_sell_when_highlimit_open(context):
    g.strategys['微盘策略'].sell_when_highlimit_open(context)
    if g.strategys['微盘策略'].is_stoplost_or_highlimit:
        g.strategys['微盘策略'].select(context)
        g.strategys['微盘策略'].adjustwithnoRM(context)
        g.strategys['微盘策略'].is_stoplost_or_highlimit = False


# 全天策略
def adjust_qt_strategy(context):
    g.strategys["全天候策略"].adjust(context)
