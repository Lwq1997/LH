# 克隆自聚宽文章：https://www.joinquant.com/post/43965
# 标题：有赚就好-小市值剥头皮改进 2年3.86%回撤
# 作者：CMA

import talib
import numpy as np
import pandas as pd


def initialize(context):
    # 开启异步订单处理模式
    set_option('async_order', True)
    set_option('use_real_price', True)
    # set_option("avoid_future_data", True)
    # set_option("t0_mode", True)
    # set_slippage(FixedSlippage(0.02))
    set_slippage(PriceRelatedSlippage(0.002))  # (比如0.2%, 交易时加减当时价格的0.1%)

    set_commission(PerTrade(buy_cost=0.0001, sell_cost=0.0011, min_cost=0))
    set_benchmark('399303.XSHE')
    g.trade_today = True
    g.choice = 500
    g.amount = 5
    g.muster = []
    g.bucket = []
    g.summit = {}
    # log.set_level('order', 'warning')
    log.set_level('order', 'debug')
    # run_daily(sell, time='9:30', reference_security='399303.XSHE')
    # run_daily(buy, time='10:30', reference_security='399303.XSHE')

    run_daily(sell_buy, time='9:30', reference_security='399303.XSHE')


def sell_buy(context):
    sell(context)
    buy(context)


# 2-1 过滤停牌股票
def filter_paused_stock(stock_list):
    current_data = get_current_data()
    return [stock for stock in stock_list if not current_data[stock].paused]


# 2-2 过滤ST及其他具有退市标签的股票
def filter_st_stock(stock_list):
    current_data = get_current_data()
    return [stock for stock in stock_list
            if not current_data[stock].is_st
            and 'ST' not in current_data[stock].name
            and '*' not in current_data[stock].name
            and '退' not in current_data[stock].name]


# 2-3 过滤科创北交股票 30为创业板
def filter_kcbj_stock(stock_list):
    for stock in stock_list[:]:
        if stock[0] == '4' or stock[0] == '8' or stock[:2] == '68' or stock[:2] == '30':
            stock_list.remove(stock)
    return stock_list


# 2-4 过滤涨停的股票
def filter_limitup_stock(context, stock_list):
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    current_data = get_current_data()
    return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
            or last_prices[stock][-1] < current_data[stock].high_limit]


# 2-5 过滤跌停的股票
def filter_limitdown_stock(context, stock_list):
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    current_data = get_current_data()
    return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
            or last_prices[stock][-1] > current_data[stock].low_limit]


# 2-6 过滤次新股
def filter_new_stock(context, stock_list):
    yesterday = context.previous_date
    return [stock for stock in stock_list if
            not yesterday - get_security_info(stock).start_date < datetime.timedelta(days=375)]


def before_trading_start(context):
    g.trade_today = True
    log.info('------------------------------------------------------------')
    fundamentals_data = get_fundamentals(
        query(valuation.code, valuation.market_cap).order_by(valuation.market_cap.asc()).limit(g.choice))
    stocks = list(fundamentals_data['code'])
    current_data = get_current_data()
    g.muster = [s for s in stocks if not current_data[s].paused
                and not current_data[s].is_st
                and 'ST' not in current_data[s].name
                and '*' not in current_data[s].name
                and '退' not in current_data[s].name
                and current_data[s].low_limit < current_data[s].day_open < current_data[s].high_limit]
    g.muster = filter_paused_stock(g.muster)
    g.muster = filter_st_stock(g.muster)
    g.muster = filter_kcbj_stock(g.muster)
    g.muster = filter_limitup_stock(context, g.muster)
    g.muster = filter_limitdown_stock(context, g.muster)

    # 计算'399303.XSHE' ma
    s = '399303.XSHE'
    closes_z = history(120, '1d', 'close', s)[s]
    # five_day_avg = history(5, '1d', 'close', s)[s].mean()
    record(price=closes_z[-1],
           ma5=closes_z[-5:].mean(),
           ma10=closes_z[-10:].mean(),
           ma20=closes_z[-20:].mean(),
           ma60=closes_z[-60:].mean())

    # if closes_z[-1] < closes_z[-3:].mean():
    #    g.trade_today = False
    # his.values.flatten()
    # if closes_z[-1] < talib.EMA(closes_z.values, timeperiod=3)[-1]:
    #     g.trade_today = False


def sell(context):
    for s in context.portfolio.positions:
        if context.portfolio.positions[s].closeable_amount > 0:
            print("sell:" + s)
            order_target(s, 0)


def buy(context):
    data_today = get_current_data()
    s = '399303.XSHE'

    # prev_close = get_price(s, count=1, end_date=context.previous_date).iloc[-1]['close']
    # if (data_today[s].day_open-prev_close)/prev_close < -0.005:
    # #if not g.trade_today:
    #     print('dont trade today！')
    #     return

    # 昨日收盘价大于五日收盘均价 且 昨日最低大于今天开盘价，卖出是第二天无条件卖出。
    buy_stock = []
    for s in g.muster:
        if len(context.portfolio.positions) + len(buy_stock) >= g.amount:
            break
        if (history(5, '1d', 'paused', s).max().values[0] == 0):  # 过去5天没有停盘
            # 过滤A杀
            low = history(4, '1d', 'low', s).min().values[0]  # 过去4天最低价格
            high = history(4, '1d', 'high', s).max().values[0]  # 过去4天最高价格
            precent = (high - low) / low * 100  # 过去4天波动
            if (precent <= 10):
                open_price_today = data_today[s].day_open
                prev_close = get_price(s, count=1, end_date=context.previous_date).iloc[-1]['close']
                # five_day_avg = history(5, '1d', 'close', s)[s].mean()
                his = history(60, '1d', 'close', s)
                ema = talib.EMA(his.values.flatten(), timeperiod=5)[-1]
                if (prev_close > ema):  # 昨日的收盘价大于5日移动平均价格
                    if (get_price(s, count=1, end_date=context.previous_date).iloc[-1]['low'] > open_price_today):
                        buy_stock.append(s)
                        # 今天开盘价小于昨日最低价

    # available_slots = g.amount - len(context.portfolio.positions)
    # if available_slots <= 0:
    #    print("no position")
    #    return
    if len(buy_stock) == 0:
        return
    cash = context.portfolio.cash  # use half every day
    allocation = cash / len(buy_stock)

    for s in buy_stock:
        open_price_today = data_today[s].day_open
        order(s, int(allocation / open_price_today))

        # if (get_price(s, count=1, end_date=context.previous_date).iloc[-1]['low'] > open_price_today) & (prev_close > five_day_avg):
        #    print("buy:" +s)
        #    order(s, int(allocation/open_price_today))
        #    最后一段什么意思？难道是昨收大于今开且昨收大于五日均价



