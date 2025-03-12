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
from UtilsToolClass import UtilsToolClass
from SBGK_Strategy_V3 import SBGK_Strategy_V3
from RZQ_Strategy_V3 import RZQ_Strategy_V3
from SBDK_Strategy_V3 import SBDK_Strategy_V3

from Strategy import Strategy


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
    log.set_level('strategy', 'info')
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
    g.portfolio_value_proportion = [0, 0, 0, 1]

    # 创建策略实例
    # 初始化策略子账户 subportfolios
    set_subportfolios([
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[0], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[1], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[2], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[3], 'stock'),
    ])

    # 是否发送微信消息，回测环境不发送，模拟环境发送
    context.is_send_wx_message = 0

    params = {
        'max_hold_count': 100,  # 最大持股数
        'max_select_count': 100,  # 最大输出选股数
    }
    sbgk_strategy = SBGK_Strategy_V3(context, subportfolio_index=0, name='首板高开', params=params)
    g.strategys[sbgk_strategy.name] = sbgk_strategy

    params = {
        'max_hold_count': 100,  # 最大持股数
        'max_select_count': 100,  # 最大输出选股数
    }
    rzq_strategy = RZQ_Strategy_V3(context, subportfolio_index=1, name='弱转强', params=params)
    g.strategys[rzq_strategy.name] = rzq_strategy

    params = {
        'max_hold_count': 100,  # 最大持股数
        'max_select_count': 100,  # 最大输出选股数
    }
    sbdk_strategy = SBDK_Strategy_V3(context, subportfolio_index=2, name='首板低开', params=params)
    g.strategys[sbdk_strategy.name] = sbdk_strategy

    params = {
        'max_hold_count': 100,  # 最大持股数
        'max_select_count': 100,  # 最大输出选股数
    }
    total_strategy = Strategy(context, subportfolio_index=3, name='统筹交易策略', params=params)
    g.strategys[total_strategy.name] = total_strategy


# 模拟盘在每天的交易时间结束后会休眠，第二天开盘时会恢复，如果在恢复时发现代码已经发生了修改，则会在恢复时执行这个函数。 具体的使用场景：可以利用这个函数修改一些模拟盘的数据。
def after_code_changed(context):  # 输出运行时间
    log.info('函数运行时间(after_code_changed)：' + str(context.current_dt.time()))

    # 是否发送微信消息，回测环境不发送，模拟环境发送
    context.is_send_wx_message = 0

    unschedule_all()  # 取消所有定时运行

    run_daily(prepare_stock_list, time='09:00')

    if g.portfolio_value_proportion[3] > 0:
        # 选股
        run_daily(total_select, time='09:26')
        run_daily(total_buy, time='09:27')
        run_daily(total_sell, time='11:25')
        run_daily(total_sell, time='14:50')


def prepare_stock_list(context):
    utilstool = UtilsToolClass()
    utilstool.name = '总策略'

    # 文本日期
    date = context.previous_date

    date_2, date_1, date = get_trade_days(end_date=date, count=3)

    # 初始列表
    initial_list = utilstool.stockpool(context, is_filter_highlimit=False,
                                       is_filter_lowlimit=False, is_updown_limit=False)

    # 昨日涨停
    yes_hl_list = utilstool.get_hl_stock(context, initial_list, date)
    # 前日曾涨停过（包含涨停+涨停炸板）
    hl1_list = utilstool.get_ever_hl_stock(context, initial_list, date_1)
    # 前前日曾涨停过（包含涨停+涨停炸板）
    hl2_list = utilstool.get_ever_hl_stock(context, initial_list, date_2)
    # 合并 hl1_list 和 hl2_list 为一个集合，用于快速查找需要剔除的元素
    elements_to_remove = set(hl1_list + hl2_list)
    # log.info('initial_list:', '001287.XSHE' in initial_list)
    # log.info('yes_hl_list:', '001287.XSHE' in yes_hl_list)
    # log.info('hl1_list:', '001287.XSHE' in hl1_list)
    # log.info('hl2_list:', '001287.XSHE' in hl2_list)
    # 昨日涨停，但是前2天都没有涨停过，真昨日首板
    context.yes_first_hl_list = [stock for stock in yes_hl_list if stock not in elements_to_remove]
    # 昨日涨停，但是前1天都没有涨停过
    context.yes_no_first_hl_list = [stock for stock in yes_hl_list if stock not in hl1_list]

    # 昨日曾涨停炸板
    h1_list = utilstool.get_ever_hl_stock2(context, initial_list, date)
    # 上上个交易日涨停
    elements_to_remove2 = utilstool.get_hl_stock(context, initial_list, date_1)

    # 过滤上上个交易日涨停、曾涨停
    context.yes_first_no_hl_list = [stock for stock in h1_list if stock not in elements_to_remove2]


def total_select(context):
    g.strategys['弱转强'].select(context)
    g.strategys['首板高开'].select(context)
    g.strategys['首板低开'].select(context)
    g.strategys['统筹交易策略'].select_list = set(
        g.strategys['弱转强'].select_list
        + g.strategys['首板高开'].select_list
        + g.strategys['首板低开'].select_list
    )


def total_buy(context):
    g.strategys['统筹交易策略'].specialBuy(context, split=3)


def total_sell(context):
    g.strategys['统筹交易策略'].specialSell(context)
