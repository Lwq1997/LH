'''
偷鸡摸狗5.0
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
from JSG2_Strategy import JSG2_Strategy
from All_Day2_Strategy import All_Day2_Strategy
from Rotation_ETF_Strategy import Rotation_ETF_Strategy
from PJ_Strategy import PJ_Strategy
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

    # 持久变量
    g.strategys = {}
    # 子账户 分仓
    g.risk_free_rate = 0.03  # 无风险收益率
    g.rebalancing = 3  # 每个季度调仓一次（关闭调仓）
    g.month = 0
    g.strategys_values = pd.DataFrame(
        columns=["s1", "s2", "s3", "s4"]
    )  # 策略数量必须保持一致
    g.portfolio_value_proportion = [0, 0.5, 0.3, 0.15, 0.05]

    # 创建策略实例
    # 初始化策略子账户 subportfolios
    set_subportfolios([
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[0], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[1], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[2], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[3], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[4], 'stock'),
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
    jsg_strategy = JSG2_Strategy(context, subportfolio_index=1, name='搅屎棍策略', params=params)
    g.strategys[jsg_strategy.name] = jsg_strategy

    params = {
    }
    all_day_strategy = All_Day2_Strategy(context, subportfolio_index=2, name='全天候策略', params=params)
    g.strategys[all_day_strategy.name] = all_day_strategy

    params = {
        'max_hold_count': 1
    }
    rotation_etf_strategy = Rotation_ETF_Strategy(context, subportfolio_index=3, name='核心资产轮动策略', params=params)
    g.strategys[rotation_etf_strategy.name] = rotation_etf_strategy

    params = {
        'max_hold_count': 1
    }
    pj_strategy = PJ_Strategy(context, subportfolio_index=4, name='破净策略', params=params)
    g.strategys[pj_strategy.name] = pj_strategy


# 模拟盘在每天的交易时间结束后会休眠，第二天开盘时会恢复，如果在恢复时发现代码已经发生了修改，则会在恢复时执行这个函数。 具体的使用场景：可以利用这个函数修改一些模拟盘的数据。
def after_code_changed(context):  # 输出运行时间
    log.info('函数运行时间(after_code_changed)：' + str(context.current_dt.time()))

    # 是否发送微信消息，回测环境不发送，模拟环境发送
    context.is_send_wx_message = 0

    unschedule_all()  # 取消所有定时运行

    # 计算子策略净值、策略仓位动态调整
    run_daily(get_strategys_values, "18:00")
    run_monthly(calculate_optimal_weights, 1, "19:00")

    # 子策略执行计划
    if g.portfolio_value_proportion[1] > 0:
        run_daily(jsg_prepare, "7:00")
        run_weekly(jsg_select, 1, "7:30")
        run_weekly(jsg_open_market, 1, "9:30")
        run_weekly(jsg_adjust, 1, "9:31")
        run_daily(jsg_check, "14:50")

    if g.portfolio_value_proportion[2] > 0:
        run_monthly(all_day_adjust, 1, "9:40")

    if g.portfolio_value_proportion[3] > 0:
        run_daily(rotation_etf_select, "7:30")
        run_daily(rotation_etf_adjust, "9:32")

    if g.portfolio_value_proportion[4] > 0:  # 如果核心资产轮动策略分配了资金
        run_daily(pj_prepare, "7:00")
        run_daily(pj_select, "7:30")
        run_daily(pj_adjust, "9:31")
        run_daily(pj_check, "14:00")  # 每天14:00检查破净策略
        run_daily(pj_check, "14:50")  # 每天14:50检查破净策略

    run_daily(after_market_close, 'after_close')


# 每日获取子策略净值
def get_strategys_values(context):
    df = g.strategys_values
    data = dict(
        zip(
            df.columns,
            [context.subportfolios[i + 1].total_value for i in range(len(df.columns))],
        )
    )
    df.loc[len(df)] = data
    if len(df) > 250:
        df = df.drop(0)
    g.strategys_values = df


# 计算最高夏普配比
def calculate_optimal_weights(context, alpha=0.5):
    current_weights = [
        round(context.subportfolios[i].total_value / context.portfolio.total_value, 3)
        for i in range(len(g.portfolio_value_proportion))
    ]
    log.info("目前仓位比例:", current_weights)
    df = g.strategys_values
    g.month += 1
    if len(df) < 250 or not g.month % g.rebalancing == 0:
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


def jsg_prepare(context):
    g.strategys["搅屎棍策略"].day_prepare(context)


def jsg_select(context):
    g.strategys["搅屎棍策略"].select(context)


def jsg_adjust(context):
    g.strategys["搅屎棍策略"].adjustwithnoRMBalance(context)


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
    g.strategys["核心资产轮动策略"].adjustwithnoRMBalance(context)


def pj_prepare(context):
    g.strategys["破净策略"].day_prepare(context)


def pj_select(context):
    g.strategys["破净策略"].select(context)


def pj_adjust(context):
    g.strategys["破净策略"].adjustwithnoRMBalance(context)


def pj_check(context):
    g.strategys["破净策略"].sell_when_highlimit_open(context)


def after_market_close(context):
    g.strategys['搅屎棍策略'].after_market_close(context)
    g.strategys['全天候策略'].after_market_close(context)
    g.strategys['核心资产轮动策略'].after_market_close(context)
    g.strategys['破净策略'].after_market_close(context)
