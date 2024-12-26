'''
多策略分子账户并行
用到的策略：
蚂蚁量化,东哥：白马股攻防转换策略（sbgk策略）
linlin2018，ZLH：低波全天候策略（外盘ETF策略）
@荒唐的方糖大佬:国九条小市值（sbgk）（还可以改进）
'''

# 导入函数库
# -*- coding: utf-8 -*-
# 如果你的文件包含中文, 请在文件的第一行使用上面的语句指定你的文件编码

from kuanke.user_space_api import *
from kuanke.wizard import *
from jqdata import *
from jqfactor import *
from jqlib.technical_analysis import *
from 策略合集.SBGK_Strategy import SBGK_Strategy

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
    set_slippage(PriceRelatedSlippage(0), type='stock')

    # 持久变量
    g.strategys = {}
    # 子账户 分仓
    g.portfolio_value_proportion = [0.33, 0.33, 0.34]

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
    sbgk_strategy = SBGK_Strategy(context, subportfolio_index=0, name='首板放量高开-低开-弱转强', params=params)
    g.strategys[sbgk_strategy.name] = sbgk_strategy

    params = {
        'max_hold_count': 100,  # 最大持股数
        'max_select_count': 100,  # 最大输出选股数
    }
    sbgk_strategy = SBGK_Strategy(context, subportfolio_index=0, name='首板放量高开-低开-弱转强', params=params)
    g.strategys[sbgk_strategy.name] = sbgk_strategy

    params = {
        'max_hold_count': 100,  # 最大持股数
        'max_select_count': 100,  # 最大输出选股数
    }
    sbgk_strategy = SBGK_Strategy(context, subportfolio_index=0, name='首板放量高开-低开-弱转强', params=params)
    g.strategys[sbgk_strategy.name] = sbgk_strategy

# 模拟盘在每天的交易时间结束后会休眠，第二天开盘时会恢复，如果在恢复时发现代码已经发生了修改，则会在恢复时执行这个函数。 具体的使用场景：可以利用这个函数修改一些模拟盘的数据。
def after_code_changed(context):  # 输出运行时间
    log.info('函数运行时间(after_code_changed)：' + str(context.current_dt.time()))

    g.n_days_limit_up_list = []  # 重新初始化列表

    # 是否发送微信消息，回测环境不发送，模拟环境发送
    context.is_send_wx_message = 0

    unschedule_all()  # 取消所有定时运行

    if g.portfolio_value_proportion[0] > 0:
        # 选股
        run_daily(sbgk_select, time='09:26')
        # 买入
        run_daily(sbgk_buy, time='9:26')
        # 卖出
        run_daily(sbgk_sell, time='11:25')
        run_daily(sbgk_sell, time='14:50')
        # 收盘
        run_daily(sbgk_after_market_close, 'after_close')


def sbgk_select(context):
    g.strategys['首板放量高开-低开-弱转强'].select(context)


def sbgk_buy(context):
    g.strategys['首板放量高开-低开-弱转强'].adjustwithnoRM(context, only_buy=True)


def sbgk_sell(context):
    g.strategys['首板放量高开-低开-弱转强'].specialSell(context)


def sbgk_after_market_close(context):
    g.strategys['首板放量高开-低开-弱转强'].after_market_close(context)
