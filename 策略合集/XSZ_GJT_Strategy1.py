# 克隆自聚宽文章：https://www.joinquant.com/post/53840
# 标题：国九小市值-11年狂飙近2000倍，绝无未来函数
# 作者：乐仔是只小猫咪

from jqdata import *
from jqfactor import *
import numpy as np
import pandas as pd
from datetime import time


def initialize(context):
    set_option('avoid_future_data', True)
    set_benchmark('000001.XSHG')
    set_option('use_real_price', True)
    set_slippage(FixedSlippage(3 / 10000))
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001, open_commission=2.5 / 10000, close_commission=2.5 / 10000,
                             close_today_commission=0, min_commission=5), type='stock')
    log.set_level('order', 'error')
    log.set_level('system', 'error')
    log.set_level('strategy', 'debug')

    g.no_trading_today_signal = False
    g.pass_april = True
    g.run_stoploss = True
    g.hold_list = []
    g.yesterday_HL_list = []
    g.target_list = []
    g.not_buy_again = []
    g.stock_num = 7
    g.up_price = 100
    g.reason_to_sell = ''
    g.stoploss_strategy = 3
    g.stoploss_limit = 0.88
    g.stoploss_market = 0.94
    g.HV_control = False
    g.HV_duration = 120
    g.HV_ratio = 0.9

    run_daily(prepare_stock_list, '9:05')
    run_weekly(weekly_adjustment, 1, '14:56')
    # run_daily(sell_stocks, time='14:57')
    run_daily(trade_afternoon, time='14:58')
    run_daily(close_account, '14:59')
    run_weekly(print_position_info, 5, time='15:10')
    run_daily(rec, '15:30')


def rec(context):
    stock_count = len(context.portfolio.positions)
    log.info(f'当前持股数 {stock_count}')
    record(总持股数=stock_count)

    position_value_ratio = context.portfolio.positions_value / context.portfolio.total_value
    log.info(f'当前仓位 {position_value_ratio:.2%}')
    record(当前仓位=int(position_value_ratio * 100), full=100, empty=0)


def prepare_stock_list(context):
    g.hold_list = []
    for position in list(context.portfolio.positions.values()):
        stock = position.security
        g.hold_list.append(stock)
    if g.hold_list != []:
        df = get_price(g.hold_list, end_date=context.previous_date, frequency='daily',
                       fields=['close', 'high_limit', 'low_limit'], count=1, panel=False, fill_paused=False)
        df = df[df['close'] == df['high_limit']]
        g.yesterday_HL_list = list(df.code)
    else:
        g.yesterday_HL_list = []
    g.no_trading_today_signal = today_is_between(context)


def get_stock_list(context):
    final_list = []
    MKT_index = '399101.XSHE'
    initial_list = get_index_stocks(MKT_index)
    initial_list = filter_new_stock(context, initial_list)
    initial_list = filter_kcbj_stock(initial_list)
    initial_list = filter_st_stock(initial_list)
    initial_list = filter_paused_stock(initial_list)
    initial_list = filter_limitup_stock(context, initial_list)
    initial_list = filter_limitdown_stock(context, initial_list)

    q = query(valuation.code, indicator.eps).filter(valuation.code.in_(initial_list)).order_by(
        valuation.market_cap.asc())
    df = get_fundamentals(q)
    stock_list = list(df.code)[:100]
    final_list = stock_list[:2 * g.stock_num]
    return final_list


def weekly_adjustment(context):
    if not g.no_trading_today_signal:
        g.not_buy_again = []
        g.target_list = get_stock_list(context)
        target_list = g.target_list[:g.stock_num]

        for stock in g.hold_list:
            if (stock not in target_list) and (stock not in g.yesterday_HL_list):
                position = context.portfolio.positions[stock]
                close_position(position)

        buy_security(context, target_list)
        for position in list(context.portfolio.positions.values()):
            g.not_buy_again.append(position.security)


def check_limit_up(context):
    if g.yesterday_HL_list:
        for stock in g.yesterday_HL_list:
            current_data = get_price(stock, end_date=context.current_dt, frequency='1m', fields=['close', 'high_limit'],
                                     skip_paused=False, fq='pre', count=1, panel=False, fill_paused=True)
            if current_data.iloc[0, 0] < current_data.iloc[0, 1]:
                position = context.portfolio.positions[stock]
                close_position(position)
                g.reason_to_sell = 'limitup'


def check_remain_amount(context):
    if g.reason_to_sell == 'limitup':
        g.hold_list = [pos.security for pos in context.portfolio.positions.values()]
        if len(g.hold_list) < g.stock_num:
            target_list = [s for s in g.target_list if s not in g.not_buy_again][:g.stock_num]
            buy_security(context, target_list)
        g.reason_to_sell = ''


def trade_afternoon(context):
    if not g.no_trading_today_signal:
        check_limit_up(context)
        if g.HV_control:
            check_high_volume(context)
        check_remain_amount(context)


def sell_stocks(context):
    if g.run_stoploss:
        stock_df = get_price(security=get_index_stocks('399101.XSHE'), end_date=context.previous_date,
                             frequency='daily', fields=['close', 'open'], count=1, panel=False)
        down_ratio = (stock_df['close'] / stock_df['open']).mean()

        if g.stoploss_strategy == 3 and down_ratio <= g.stoploss_market:
            g.reason_to_sell = 'stoploss'
            for stock in context.portfolio.positions.keys():
                order_target_value(stock, 0)
        else:
            for stock in context.portfolio.positions.keys():
                if context.portfolio.positions[stock].price < context.portfolio.positions[
                    stock].avg_cost * g.stoploss_limit:
                    order_target_value(stock, 0)
                    g.reason_to_sell = 'stoploss'


def check_high_volume(context):
    current_data = get_current_data()
    for stock in context.portfolio.positions:
        if not (current_data[stock].paused or
                current_data[stock].last_price == current_data[stock].high_limit or
                context.portfolio.positions[stock].closeable_amount == 0):
            df_volume = get_bars(stock, count=g.HV_duration, unit='1d', fields=['volume'], include_now=True, df=True)

            if df_volume['volume'].values[-1] > g.HV_ratio * df_volume['volume'].values.max():
                close_position(context.portfolio.positions[stock])


def filter_paused_stock(stock_list):
    current_data = get_current_data()
    return [stock for stock in stock_list if not current_data[stock].paused]


def filter_st_stock(stock_list):
    current_data = get_current_data()
    return [stock for stock in stock_list
            if not current_data[stock].is_st
            and 'ST' not in current_data[stock].name
            and '*' not in current_data[stock].name
            and '退' not in current_data[stock].name]


def filter_kcbj_stock(stock_list):
    return [s for s in stock_list if s[0] not in ['4', '8'] and s[:2] != '68']


def filter_limitup_stock(context, stock_list):
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    current_data = get_current_data()
    return [s for s in stock_list if
            s in context.portfolio.positions or last_prices[s][-1] < current_data[s].high_limit]


def filter_limitdown_stock(context, stock_list):
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    current_data = get_current_data()
    return [s for s in stock_list if s in context.portfolio.positions or last_prices[s][-1] > current_data[s].low_limit]


def filter_new_stock(context, stock_list):
    yesterday = context.previous_date
    return [s for s in stock_list if not yesterday - get_security_info(s).start_date < datetime.timedelta(days=375)]


def filter_not_buy_again(stock_list):
    return [s for s in stock_list if s not in g.not_buy_again]


def order_target_value_(security, value):
    return order_target_value(security, value)


def open_position(security, value):
    order = order_target_value_(security, value)
    return order.filled > 0 if order else False


def close_position(position):
    order = order_target_value_(position.security, 0)
    return order.filled == position.total_amount if order else False


def buy_security(context, target_list):
    position_count = len(context.portfolio.positions)
    target_num = len(target_list)
    if target_num > position_count:
        value = context.portfolio.cash / (target_num - position_count)
        for stock in target_list:
            if context.portfolio.positions[stock].total_amount == 0:
                if open_position(stock, value):
                    g.not_buy_again.append(stock)
                    if len(context.portfolio.positions) == target_num:
                        break


def today_is_between(context):
    today = context.current_dt.strftime('%m-%d')
    return (('04-01' <= today <= '04-30') or ('01-01' <= today <= '01-30')) if g.pass_april else False


def close_account(context):
    if g.no_trading_today_signal and g.hold_list:
        for stock in g.hold_list:
            close_position(context.portfolio.positions[stock])


def print_position_info(context):
    for position in context.portfolio.positions.values():
        print(f'代码:{position.security}')
        print(f'成本价:{position.avg_cost:.2f}')
        print(f'现价:{position.price}')
        print(f'收益率:{100 * (position.price / position.avg_cost - 1):.2f}%')
        print(f'持仓(股):{position.total_amount}')
        print(f'市值:{position.value:.2f}')
        print('-' * 40)
    print('-' * 50 + '分割线' + '-' * 50)
