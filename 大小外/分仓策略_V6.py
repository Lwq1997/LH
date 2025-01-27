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
from BMZH_Strategy import BMZH_Strategy
from WPETF_Strategy import WPETF_Strategy
from XSZ_GJT_Strategy import XSZ_GJT_Strategy
from scipy.optimize import minimize


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
    g.portfolio_value_proportion = [0, 0.35, 0.15, 0.50]
    # 全局变量
    # g.strategys = {}
    g.risk_free_rate = 0.03  # 无风险收益率
    g.is_balanced = False  # 是否开启自动调仓
    g.rebalancing = 3  # 每个季度调仓一次
    g.month = 0  # 记录时间
    g.strategys_values = pd.DataFrame(
        columns=["s1", "s2", "s3"]
    )  # 子策略净值
    g.strategys_days = 250  # 取子策略净值最近250个交易日
    g.after_factor = [1, 1, 1]  # 后复权因子
    # g.portfolio_value_proportion = [0, 0.3, 0.6, 0.1]

    # 创建策略实例
    # 初始化策略子账户 subportfolios
    set_subportfolios([
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[0], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[1], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[2], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[3], 'stock'),
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
    bmzh_strategy = BMZH_Strategy(context, subportfolio_index=1, name='白马股攻防转换策略', params=params)
    g.strategys[bmzh_strategy.name] = bmzh_strategy

    params = {
        'max_hold_count': 1,  # 最大持股数
        'max_select_count': 2,  # 最大输出选股数
    }
    # ETF 策略，第二个仓
    wpetf_strategy = WPETF_Strategy(context, subportfolio_index=2, name='外盘ETF轮动策略', params=params)
    g.strategys[wpetf_strategy.name] = wpetf_strategy

    params = {
        'max_hold_count': 3,  # 最大持股数
        'max_select_count': 10,  # 最大输出选股数
        'use_empty_month': True,  # 是否在指定月份空仓
        'empty_month': [1, 4]  # 指定空仓的月份列表
    }
    # 小世值，第三个仓
    xszgjt_strategy = XSZ_GJT_Strategy(context, subportfolio_index=3, name='国九条小市值策略', params=params)
    g.strategys[xszgjt_strategy.name] = xszgjt_strategy


# 模拟盘在每天的交易时间结束后会休眠，第二天开盘时会恢复，如果在恢复时发现代码已经发生了修改，则会在恢复时执行这个函数。 具体的使用场景：可以利用这个函数修改一些模拟盘的数据。
def after_code_changed(context):  # 输出运行时间
    log.info('函数运行时间(after_code_changed)：' + str(context.current_dt.time()))

    # 是否发送微信消息，回测环境不发送，模拟环境发送
    context.is_send_wx_message = 0

    unschedule_all()  # 取消所有定时运行

    # 计算子策略净值、策略仓位动态调整
    run_daily(get_strategys_values, "18:00")
    run_monthly(calculate_optimal_weights, 1, "19:00")

    # 执行计划
    # 选股函数--Select：白马和 ETF 分开使用
    # 执行函数--adjust：白马和 ETF 轮动共用一个
    # # 白马，按月运行 TODO
    if g.portfolio_value_proportion[1] > 0:
        run_monthly(bmzh_select, 1, time='7:40')  # 阅读完成，测试完成
        run_monthly(bmzh_adjust, 1, time='09:35')  # 阅读完成，测试完成
        run_daily(bmzh_after_market_close, 'after_close')
    #
    # # ETF轮动，按天运行
    if g.portfolio_value_proportion[2] > 0:
        run_daily(wpetf_select, time='7:42')  # 阅读完成，测试完成
        run_daily(wpetf_adjust, time='09:35')  # 阅读完成，测试完成
        run_daily(wpetf_after_market_close, 'after_close')

    # # 小市值，按天/周运行
    if g.portfolio_value_proportion[3] > 0:
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


# 每日获取子策略净值
def get_strategys_values(context):
    df = g.strategys_values.copy()
    data = dict(
        zip(
            df.columns,
            [
                context.subportfolios[i + 1].total_value * g.after_factor[i]
                for i in range(len(df.columns))
            ],
        )
    )
    df.loc[len(df)] = data
    if len(df) > g.strategys_days:
        df = df.drop(0)
    g.strategys_values = df


# 计算最高夏普配比
def calculate_optimal_weights(context, alpha=0.2):
    current_weights = [
        round(context.subportfolios[i].total_value / context.portfolio.total_value, 3)
        for i in range(len(g.portfolio_value_proportion))
    ]
    log.info("目前仓位比例:", current_weights)
    df = g.strategys_values
    g.month += 1
    if len(df) < g.strategys_days or not g.month % g.rebalancing == 0:
        return

    # 计算每个策略的收益率
    returns = df.pct_change().dropna()

    # 计算每个策略的年化收益率
    annualized_returns = returns.mean() * 252

    # 计算协方差矩阵
    cov_matrix = returns.cov() * 252

    # 定义目标函数：负波动率调整后的夏普比率
    def negative_vasr(weights):
        portfolio_return = np.dot(weights, annualized_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - g.risk_free_rate) / portfolio_volatility
        vasr = sharpe_ratio / (1 + alpha * portfolio_volatility)
        return -vasr

    # 约束条件：权重之和为1
    constraints = [
        {"type": "eq", "fun": lambda x: np.sum(x) - 1},
        {"type": "ineq", "fun": lambda x: x - 0.05},  # 确保每个权重都大于等于0.05
    ]

    # 添加约束：每个策略前后配比之差不超过10%
    last_best_weights = g.portfolio_value_proportion[1:]  # 去掉第一个0
    constraints.append(
        {"type": "ineq", "fun": lambda x: 0.1 - np.abs(x - last_best_weights)}
    )

    # 添加约束：单个策略最大比重不超过50%
    constraints.append({"type": "ineq", "fun": lambda x: 0.5 - x})

    # 权重的初始猜测
    num_strategies = len(returns.columns)
    initial_weights = np.array([1 / num_strategies] * num_strategies)
    initial_weights = np.maximum(initial_weights, 0.05)  # 确保初始权重符合最低配比要求

    # 优化问题
    result = minimize(
        negative_vasr, initial_weights, method="SLSQP", constraints=constraints
    )

    # 输出最佳权重
    best_weights = result.x.tolist()
    g.portfolio_value_proportion = [0] + best_weights
    log.info("最佳权重:", [round(i, 3) for i in best_weights])


# 选股
def bmzh_select(context):
    g.strategys['白马股攻防转换策略'].select(context)


# 交易
def bmzh_adjust(context):
    g.strategys['白马股攻防转换策略'].adjustwithnoRMBalance(context)


# 收盘统计
def bmzh_after_market_close(context):
    g.strategys['白马股攻防转换策略'].after_market_close(context)


def wpetf_select(context):
    g.strategys['外盘ETF轮动策略'].select(context)


def wpetf_adjust(context):
    g.strategys['外盘ETF轮动策略'].adjustwithnoRMBalance(context)


def wpetf_after_market_close(context):
    g.strategys['外盘ETF轮动策略'].after_market_close(context)


def xszgjt_day_prepare(context):
    g.strategys['国九条小市值策略'].day_prepare(context)


def xszgjt_select(context):
    g.strategys['国九条小市值策略'].select(context)


def xszgjt_adjust(context):
    g.strategys['国九条小市值策略'].clear_append_buy_dict(context)
    g.strategys['国九条小市值策略'].adjustwithnoRMBalance(context)


def xszgjt_open_market(context):
    g.strategys['国九条小市值策略'].close_for_empty_month(context)
    g.strategys['国九条小市值策略'].close_for_stoplost(context)


def xszgjt_sell_when_highlimit_open(context):
    g.strategys['国九条小市值策略'].sell_when_highlimit_open(context)


def xszgjt_append_buy_stock(context):
    g.strategys['国九条小市值策略'].append_buy_dict(context)


def xszgjt_after_market_close(context):
    g.strategys['国九条小市值策略'].after_market_close(context)
