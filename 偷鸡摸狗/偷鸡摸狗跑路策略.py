# 克隆自聚宽文章：https://www.joinquant.com/post/55109
# 标题：瞎改了一下MarioC老师的[偷鸡摸狗跑路]策略
# 作者：奇妙的午后

from jqdata import *
from jqfactor import *
import numpy as np
import pandas as pd
import math
from datetime import datetime


# 初始化函数
def initialize(context):
    set_benchmark('000300.XSHG')  # 设置基准为沪深300
    set_option('use_real_price', True)  # 使用真实价格
    set_option("avoid_future_data", True)  # 避免未来函数
    set_slippage(FixedSlippage(0.001))  # 设置滑点
    set_order_cost(
        OrderCost(open_tax=0, close_tax=0, open_commission=0.0002, close_commission=0.0002, close_today_commission=0,
                  min_commission=5), type='fund')  # 设置交易成本
    log.set_level('system', 'error')  # 过滤日志

    # 参数设置
    context.ETF_POOL = [
        '518880.XSHG',  # 黄金ETF（大宗商品）
        '513100.XSHG',  # 纳指100（海外资产）
        '159915.XSHE',  # 创业板100（成长股）
        '510180.XSHG',  # 上证180（价值股）
    ]
    context.stock_num = 10  # 持股数量
    context.strategy_type = '跑路'  # 默认策略类型
    context.counterattack_days = 20  # 策略切换天数
    context.momentum_days = 25  # 动量计算天数
    context.days_counter = 0  # 计数器
    context.initial_days = 100  # 强制“跑路”模式的天数
    context.initial_counter = 0  # 强制“跑路”模式的计数器

    # 定时任务
    run_daily(prepare_stock_list, '9:05')  # 准备股票列表
    run_daily(trade, '9:30')  # 交易逻辑
    run_daily(stop_loss, '14:00')  # 止损逻辑


def stop_loss(context):
    now_time = context.current_dt
    num = 0

    # 处理涨停股
    if context.yesterday_HL_list:
        for stock in context.yesterday_HL_list:
            current_data = get_price(stock, end_date=now_time, frequency='1m', fields=['close', 'high_limit'],
                                     skip_paused=False, fq='pre', count=1, panel=False)
            if current_data.iloc[0, 0] < current_data.iloc[0, 1]:
                log.info(f"{stock} 涨停打开，卖出")
                close_position(context.portfolio.positions[stock])
                num += 1
            else:
                log.info(f"{stock} 涨停，继续持有")

    # 止损处理
    SS = []
    S = []
    for stock in context.hold_list:
        if stock not in context.ETF_POOL and stock in context.portfolio.positions:
            position = context.portfolio.positions[stock]
            if position.price < position.avg_cost * 0.92:
                order_target_value(stock, 0)
                log.debug(f"止损卖出 {stock}")
                num += 1
            else:
                S.append(stock)
                SS.append((position.price - position.avg_cost) / position.avg_cost)

    # 补仓逻辑
    if num >= 1 and SS:
        min_indices = np.argsort(SS)[:3]
        min_stocks = [S[i] for i in min_indices if i < len(S)]
        cash = context.portfolio.cash / len(min_stocks)
        for stock in min_stocks:
            order_value(stock, cash)
            log.debug(f"补仓 {stock}")


def prepare_stock_list(context):
    context.hold_list = list(context.portfolio.positions.keys())
    if context.hold_list:
        df = get_price(context.hold_list, end_date=context.previous_date, frequency='daily',
                       fields=['close', 'high_limit'], count=1, panel=False, fill_paused=False)
        df = df[df['close'] == df['high_limit']]
        context.yesterday_HL_list = list(df.code)
    else:
        context.yesterday_HL_list = []


def calculate_momentum(etf, days=25):
    df = attribute_history(etf, days, '1d', ['close'])
    y = np.log(df['close'].values)
    x = np.arange(len(y))
    weights = np.linspace(1, 2, len(y))  # 线性增加权重
    slope, intercept = np.polyfit(x, y, 1, w=weights)
    annualized_return = math.exp(slope * 250) - 1
    residuals = y - (slope * x + intercept)
    r_squared = 1 - (np.sum(weights * residuals ** 2) / np.sum(weights * (y - np.mean(y)) ** 2))
    return annualized_return * r_squared


def filter_stocks(context, stock_list):
    stock_list = filter_paused_stock(stock_list)
    stock_list = filter_st_stock(stock_list)
    stock_list = filter_kcbj_stock(stock_list)
    stock_list = filter_new_stock(context, stock_list)
    stock_list = filter_highprice_stock(context, stock_list)
    return stock_list


def get_blue_chip_stocks(context, stock_index='000300.XSHG'):
    current_data = get_current_data()
    stock_list = get_index_stocks(stock_index, current_data)

    # 过滤高价股、ST股、次新股和科创板/北交所股票
    stock_list = filter_stocks(context, stock_list)

    # 多维度筛选
    roic_stocks = filter_roic_stocks(context, stock_list)
    big_stocks = filter_big_stocks(context, stock_list)
    bm_stocks = filter_bm_stocks(context, stock_list)

    # 合并筛选结果并去重
    combined_stocks = list(set(roic_stocks + big_stocks + bm_stocks))

    # 按市值降序排序并返回前context.stock_num只股票
    fundamentals_query = get_fundamentals(
        query(valuation.code)
        .filter(valuation.code.in_(combined_stocks))
        .order_by(valuation.market_cap.desc())
        .limit(context.stock_num)
    )

    return list(fundamentals_query.code)


def filter_roic_stocks(context, stock_list):
    yesterday = context.previous_date
    roic_stocks = []
    for stock in stock_list:
        roic = get_factor_values(stock, 'roic_ttm', end_date=yesterday, count=1)['roic_ttm'].iloc[0, 0]
        if roic > 0.08:
            roic_stocks.append(stock)
    return roic_stocks


def filter_big_stocks(context, stock_list):
    query_result = get_fundamentals(
        query(valuation.code)
        .filter(
            valuation.code.in_(stock_list),
            valuation.pe_ratio_lyr.between(0, 30),
            valuation.ps_ratio.between(0, 8),
            valuation.pcf_ratio < 10,
            indicator.eps > 0.3,
            indicator.roe > 0.1,
            indicator.net_profit_margin > 0.1,
            indicator.gross_profit_margin > 0.3,
            indicator.inc_revenue_year_on_year > 0.25
        )
    )
    return list(query_result.code)


def filter_bm_stocks(context, stock_list):
    query_result = get_fundamentals(
        query(valuation.code)
        .filter(
            valuation.code.in_(stock_list),
            valuation.market_cap.between(100, 900),
            valuation.pb_ratio.between(0, 10),
            valuation.pcf_ratio < 4,
            indicator.eps > 0.3,
            indicator.roe > 0.2,
            indicator.net_profit_margin > 0.1,
            indicator.inc_revenue_year_on_year > 0.2,
            indicator.inc_operation_profit_year_on_year > 0.1
        )
    )
    return list(query_result.code)


def get_small_cap_stocks(context, stock_index='399101.XSHE'):
    current_data = get_current_data()
    stock_list = get_index_stocks(stock_index, current_data)

    # 过滤高价股、ST股、次新股和科创板/北交所股票
    stock_list = filter_stocks(context, stock_list)

    # 筛选小盘股
    query_result = get_fundamentals(
        query(valuation.code)
        .filter(
            valuation.code.in_(stock_list),
            indicator.roe > 0.15,
            indicator.roa > 0.10
        )
        .order_by(valuation.market_cap.asc())
        .limit(context.stock_num)
    )

    return list(query_result.code)


def trade(context):
    current_data = get_current_data()

    # 强制“跑路”模式
    if context.initial_counter < context.initial_days:
        log.info(f"强制“跑路”模式，第 {context.initial_counter + 1} 天")
        context.initial_counter += 1
        execute_rout_strategy(context)
        return

    # 策略切换逻辑
    next_strategy = None
    if context.strategy_type == '偷鸡':
        if context.days_counter < context.counterattack_days:
            context.days_counter += 1
            next_strategy = '偷鸡'
        else:
            next_strategy = '跑路'
            context.days_counter = 0
    elif context.strategy_type == '摸狗':
        if context.days_counter < context.counterattack_days:
            context.days_counter += 1
            next_strategy = '摸狗'
        else:
            next_strategy = '跑路'
            context.days_counter = 0
    else:
        target_etf = get_top_momentum_etf(context.ETF_POOL)
        if target_etf == '159915.XSHE':
            next_strategy = '偷鸡'
        elif target_etf == '510180.XSHG':
            next_strategy = '摸狗'
        else:
            next_strategy = '跑路'
        context.days_counter = 0  # 重置计数器

    # 如果下一个策略与当前策略相同，保持当前持仓
    if next_strategy == context.strategy_type:
        log.info(f"策略保持不变，继续执行 {context.strategy_type} 策略")
        return

    # 更新策略类型
    context.strategy_type = next_strategy

    # 执行对应策略
    if context.strategy_type == '偷鸡':
        execute_steal_chicken_strategy(context, current_data)
    elif context.strategy_type == '摸狗':
        execute_modog_strategy(context, current_data)
    else:
        execute_rout_strategy(context)


def clear_portfolio(context):
    for stock in list(context.portfolio.positions.keys()):
        order_target_value(stock, 0)


def get_top_momentum_etf(etf_pool):
    scores = {etf: calculate_momentum(etf) for etf in etf_pool}
    return max(scores, key=scores.get)


def execute_steal_chicken_strategy(context, current_data):
    if context.days_counter == 0:
        log.info("开始偷鸡策略")
        target_stocks = get_small_cap_stocks(context)
        adjust_portfolio(context, target_stocks)


def execute_modog_strategy(context, current_data):
    if context.days_counter == 0:
        log.info("开始摸狗策略")
        target_stocks = get_blue_chip_stocks(context)
        adjust_portfolio(context, target_stocks)


def execute_rout_strategy(context):
    log.info("开始跑路策略")
    target_etf = get_top_momentum_etf(context.ETF_POOL)

    # 如果 target_etf 是特定的 ETF，将其设置为空
    if target_etf in {'510180.XSHG', '159915.XSHE'}:
        target_etf = ''

    # 如果 target_etf 为空，直接返回，避免下单失败
    if not target_etf:
        log.info("目标 ETF 为空，跳过买入操作")
        return

    hold_stocks = list(context.portfolio.positions.keys())

    # 卖出非目标 ETF
    for stock in hold_stocks:
        if stock != target_etf:
            order_target_value(stock, 0)
            log.info(f"卖出 {stock}")

    # 买入目标 ETF
    if context.portfolio.available_cash > 10000:
        order_value(target_etf, context.portfolio.available_cash)
        log.info(f"买入 {target_etf}，金额: {context.portfolio.available_cash}")
    else:
        log.info("可用现金不足，无法买入目标 ETF")


def adjust_portfolio(context, target_stocks):
    # 卖出不在目标列表中的股票
    for stock in list(context.portfolio.positions.keys()):
        if stock not in target_stocks:
            close_position(context.portfolio.positions[stock])

    # 买入目标股票
    cash_per_stock = context.portfolio.cash / len(target_stocks)
    for stock in target_stocks:
        if stock not in context.portfolio.positions:
            order_value(stock, cash_per_stock)


def open_position(security, value):
    order = order_target_value(security, value)
    return order is not None and order.filled > 0


def close_position(position):
    order = order_target_value(position.security, 0)
    return order is not None and order.filled == order.amount


def filter_paused_stock(stock_list):
    return [stock for stock in stock_list if not get_current_data()[stock].paused]


def filter_st_stock(stock_list):
    return [stock for stock in stock_list if not get_current_data()[stock].is_st]


def filter_kcbj_stock(stock_list):
    return [stock for stock in stock_list if not (stock[0] in ['4', '8'] or stock[:2] in ['68', '3'])]


def filter_new_stock(context, stock_list):
    return [stock for stock in stock_list if (context.previous_date - get_security_info(stock).start_date).days > 375]


def filter_highprice_stock(context, stock_list):
    last_prices = history(1, '1m', 'close', stock_list)
    return [stock for stock in stock_list if last_prices[stock][-1] < 100]