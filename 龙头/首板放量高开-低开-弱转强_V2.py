'''
多策略分子账户并行
用到的策略：
蚂蚁量化,东哥：白马股攻防转换策略（dxw策略）
linlin2018，ZLH：低波全天候策略（外盘ETF策略）
@荒唐的方糖大佬:国九条小市值（dxw）（还可以改进）
'''

# 导入函数库
# -*- coding: utf-8 -*-
# 如果你的文件包含中文, 请在文件的第一行使用上面的语句指定你的文件编码

from kuanke.user_space_api import *
from kuanke.wizard import *
from jqdata import *
from jqfactor import *
from jqlib.technical_analysis import *
from 策略合集.DXW_Strategy import DXW_Strategy

import warnings
from datetime import date as dt


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

    # 持久变量
    g.strategys = {}
    # 子账户 分仓
    g.portfolio_value_proportion = [1, 0, 0]

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
        'max_hold_count': 100,  # 最大持股数
        'max_select_count': 100,  # 最大输出选股数
    }
    dxw_strategy = DXW_Strategy(context, subportfolio_index=0, name='大小外综合策略', params=params)
    g.strategys[dxw_strategy.name] = dxw_strategy


# 模拟盘在每天的交易时间结束后会休眠，第二天开盘时会恢复，如果在恢复时发现代码已经发生了修改，则会在恢复时执行这个函数。 具体的使用场景：可以利用这个函数修改一些模拟盘的数据。
def after_code_changed(context):  # 输出运行时间
    log.info('函数运行时间(after_code_changed)：' + str(context.current_dt.time()))

    g.n_days_limit_up_list = []  # 重新初始化列表

    # 是否发送微信消息，回测环境不发送，模拟环境发送
    context.is_send_wx_message = 0

    unschedule_all()  # 取消所有定时运行

    if g.portfolio_value_proportion[0] > 0:
        # 准备工作
        run_daily(dxw_day_prepare, time='7:30')
        # 选择大小外的其中一个
        run_monthly(dxw_singal, 1, time='08:00')
        # 选股
        run_weekly(dxw_select, 1, time='09:30')
        # 空仓/止损
        # run_daily(dxw_open_market, time='9:30')
        # 补仓卖出
        run_weekly(dxw_adjust, 1, time='9:30')
        # run_daily(dxw_sell_when_highlimit_open, time='11:27')
        # 非涨停出售
        run_daily(dxw_sell_when_highlimit_open, time='14:00')
        # run_daily(dxw_sell_when_highlimit_open, time='14:50')
        # 补仓买入
        run_daily(dxw_append_buy_stock, time='14:00')
        # 收盘
        run_daily(dxw_after_market_close, 'after_close')




def dxw_day_prepare(context):
    g.strategys['大小外综合策略'].day_prepare(context)


def dxw_singal(context):
    g.strategys['大小外综合策略'].singal(context)


def dxw_select(context):
    g.strategys['大小外综合策略'].select(context)


def dxw_adjust(context):
    g.strategys['大小外综合策略'].clear_append_buy_dict(context)
    g.strategys['大小外综合策略'].adjustwithnoRM(context)


def dxw_open_market(context):
    g.strategys['大小外综合策略'].close_for_empty_month(context)
    g.strategys['大小外综合策略'].close_for_stoplost(context)


def dxw_sell_when_highlimit_open(context):
    g.strategys['大小外综合策略'].sell_when_highlimit_open(context)


def dxw_append_buy_stock(context):
    g.strategys['大小外综合策略'].append_buy_dict(context)


def dxw_after_market_close(context):
    g.strategys['大小外综合策略'].after_market_close(context)
