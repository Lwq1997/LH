# 克隆自聚宽文章：https://www.joinquant.com/post/55002
# 标题：st弱转强v3 年化187% 最大回撤24 夏普6
# 作者：aqa

import pandas as pd
import numpy as np
import math
import talib as tl
import datetime as dt
from datetime import datetime
from datetime import timedelta
from jqlib.technical_analysis import *
from jqdata import *


def initialize(context):
    set_option('use_real_price', True)
    # 开启防未来函数
    set_option('avoid_future_data', True)
    # 将滑点设置为0
    set_slippage(FixedSlippage(0.01))
    # 设置交易成本万分之三，不同滑点影响可在归因分析中查看
    set_order_cost(OrderCost(open_tax=0, close_tax=0.0005, open_commission=0.0001, close_commission=0.0001,
                             close_today_commission=0, min_commission=5), type='stock')
    # ----------------Settings--------------------------------------------------/
    g.stock_num = 5
    g.today_list = []  # 当日观测股票

    # ----------------Settings--------------------------------------------------/
    run_daily(perpare, time="9:25")
    run_daily(buy, time="9:30")
    run_daily(sell, time='13:00')
    run_daily(sell, time='14:00')


def perpare(context):  # 筛选

    current_data = get_current_data()
    g.yesterday_high_dict = {}
    g.today_list = []
    stk_list = get_st(context)
    # 1 4 12月 国九
    singal = today_is_between(context)
    if singal == True:
        print(f'筛选前：{len(stk_list)}')
        stk_list = GJT_filter_stocks(stk_list)
        print(f'筛选后：{len(stk_list)}')

    stk_list = filter_stocks(context, stk_list)
    if len(stk_list) == 0:
        return
    stk_list = rzq_list(context, stk_list)
    if len(stk_list) == 0:
        return
        # 低开
    df = get_price(stk_list, end_date=context.previous_date, frequency='daily', fields=['close'], count=1, panel=False,
                   fill_paused=False, skip_paused=True).set_index('code')
    df['open_now'] = [current_data[s].day_open for s in stk_list]
    df = df[(df['open_now'] / df['close']) < 1.01]  # 低开越多风险越大，选择3个多点即可
    df = df[(df['open_now'] / df['close']) > 0.95]
    stk_list = list(df.index)
    if len(stk_list) == 0:
        return
    df = get_valuation(stk_list, start_date=context.previous_date,
                       end_date=context.previous_date,
                       fields=['turnover_ratio', 'market_cap', 'circulating_market_cap']
                       )

    df = df.sort_values(by='turnover_ratio', ascending=False)
    g.today_list = list(df.code)
    log.info(f"股池数{len(stk_list)},股池明细{stk_list}")


def sell(context):
    hold_list = list(context.portfolio.positions)
    if hold_list:
        current_data = get_current_data()
        yesterday = context.previous_date

        # 批量获取昨日涨停数据
        df_history = get_price(hold_list, end_date=yesterday, frequency='daily', fields=['close', 'high_limit'],
                               count=1, panel=False)
        df_history['avg_cost'] = [context.portfolio.positions[s].avg_cost for s in hold_list]
        df_history['price'] = [context.portfolio.positions[s].price for s in hold_list]
        df_history['today_high_limit'] = [current_data[s].high_limit for s in hold_list]
        df_history['low_limit'] = [current_data[s].low_limit for s in hold_list]
        df_history['last_price'] = [current_data[s].last_price for s in hold_list]
        # 条件1：未涨停
        cond1 = (df_history['last_price'] != df_history['today_high_limit'])
        # 条件2.1：亏损超过3%
        ret_matrix = (df_history['price'] / df_history['avg_cost'] - 1) * 100
        cond2_1 = ret_matrix < -3
        # 条件2.2：盈利超过0%
        cond2_2 = ret_matrix > 0
        # 条件2.3：昨日涨停
        cond2_3 = (df_history['close'] == df_history['high_limit'])

        # 组合卖出条件（逻辑或运算）
        sell_condition = cond1 & (cond2_1 | cond2_2 | cond2_3)
        # 生成卖出列表（过滤无效订单）
        sell_list = df_history[
            sell_condition &
            (df_history['last_price'] > df_history['low_limit'])
            ].code.tolist()

        # 批量下单
        for s in sell_list:
            order_target_value(s, 0)
            print(
                f'卖出 {s} | 成本价:{context.portfolio.positions[s].avg_cost:.2f} 现价:{context.portfolio.positions[s].price:.2f}')
            print('-' * 50)


def buy(context):
    # 1 4 12月空仓

    target = g.today_list

    hold_list = list(context.portfolio.positions)
    num = g.stock_num - len(hold_list)
    if num == 0:
        return
    target = [x for x in target if x not in hold_list][:num]
    log.info(f'最终实际购买股票:{target},最终实际购买股票数量:{num},当前持仓数量:{len(hold_list)}')
    if len(target) > 0:
        # 分配资金（等权重买入）
        value = context.portfolio.available_cash
        cash_per_stock = value / num
        current_data = get_current_data()  # 实时数据对象
        for stock in target:
            # 排除停牌和涨跌停无法交易的股票
            if current_data[stock].paused or \
                    current_data[stock].last_price == current_data[stock].low_limit or \
                    current_data[stock].last_price == current_data[stock].high_limit:
                continue
            order_value(stock, cash_per_stock)  # 按金额买入[6](@ref)
            log.info(f"买入 {stock}")


# ----------------函数群--------------------------------------------------/
def today_is_between(context):
    today = context.current_dt.strftime('%m-%d')
    if ('01-15' <= today) and (today <= '01-31'):
        return True
    elif ('04-15' <= today) and (today <= '04-31'):
        return True
    elif ('12-15' <= today) and (today <= '12-31'):
        return True
    else:
        return False


##获取所有ST股##
def get_st(context):
    yesterday = context.previous_date
    stockList = get_all_securities(types='stock', date=yesterday).index
    st_data = get_extras('is_st', stockList, count=1, end_date=yesterday)
    st_data = st_data.T
    st_data.columns = ['is_st']
    st_data = st_data[st_data['is_st'] == True]
    df = st_data.index.tolist()

    # 新增过滤逻辑
    filtered_df = [
        code for code in df
        if not (
                str(code).startswith('4') or
                str(code).startswith('8') or
                str(code).startswith('68') or
                str(code).startswith('30')
        )
    ]

    return filtered_df


##处理日期相关函数##
def get_shifted_date(date, days, days_type='T'):
    # 获取上一个自然日
    d_date = transform_date(date, 'd')
    yesterday = d_date + dt.timedelta(-1)
    # 移动days个自然日
    if days_type == 'N':
        shifted_date = yesterday + dt.timedelta(days + 1)
    # 移动days个交易日
    if days_type == 'T':
        all_trade_days = [i.strftime('%Y-%m-%d') for i in list(get_all_trade_days())]
        # 如果上一个自然日是交易日，根据其在交易日列表中的index计算平移后的交易日
        if str(yesterday) in all_trade_days:
            shifted_date = all_trade_days[all_trade_days.index(str(yesterday)) + days + 1]
        # 否则，从上一个自然日向前数，先找到最近一个交易日，再开始平移
        else:
            for i in range(100):
                last_trade_date = yesterday - dt.timedelta(i)
                if str(last_trade_date) in all_trade_days:
                    shifted_date = all_trade_days[all_trade_days.index(str(last_trade_date)) + days + 1]
                    break

    return str(shifted_date)


##处理日期相关函数##
def transform_date(date, date_type):
    if type(date) == str:
        str_date = date
        dt_date = dt.datetime.strptime(date, '%Y-%m-%d')
        d_date = dt_date.date()
    elif type(date) == dt.datetime:
        str_date = date.strftime('%Y-%m-%d')
        dt_date = date
        d_date = dt_date.date()
    elif type(date) == dt.date:
        str_date = date.strftime('%Y-%m-%d')
        dt_date = dt.datetime.strptime(str_date, '%Y-%m-%d')
        d_date = date
    dct = {'str': str_date, 'dt': dt_date, 'd': d_date}
    return dct[date_type]


##筛选不涨停##
def get_ever_hl_stock(initial_list, date):  #
    df = get_price(initial_list, end_date=date, frequency='daily', fields=['close', 'high', 'high_limit'], count=1,
                   panel=False, fill_paused=False, skip_paused=False)
    df = df.dropna()  # 去除停牌
    cd2 = df['close'] != df['high_limit']
    df = df[cd2]
    hl_list = list(df.code)
    return hl_list


##筛选出涨停的股票##
def get_hl_stock(initial_list, date):
    df = get_price(initial_list, end_date=date, frequency='daily', fields=['close', 'low', 'high_limit'], count=1,
                   panel=False, fill_paused=False, skip_paused=False)
    df = df.dropna()  # 去除停牌
    df = df[df['close'] == df['high_limit']]
    hl_list = list(df.code)
    return hl_list


##筛选昨日不涨停的股票##
def rzq_list(context, initial_list):
    # 文本日期
    date = context.previous_date  # 昨日
    date = transform_date(date, 'str')
    date_1 = get_shifted_date(date, -1, 'T')  # 前日
    date_2 = get_shifted_date(date, -2, 'T')  # 大前日
    # 昨日不涨停
    h1_list = get_ever_hl_stock(initial_list, date)
    # 前日涨停过滤
    elements_to_remove = get_hl_stock(initial_list, date_1)

    zb_list = [stock for stock in h1_list if stock in elements_to_remove]

    return zb_list


##技术指标筛选##
def filter_stocks(context, stocks):
    yesterday = context.previous_date
    df = get_price(
        stocks,
        count=11,
        frequency='1d',
        fields=['close', 'low', 'volume'],
        end_date=yesterday,
        panel=False
    ).reset_index()
    # 按股票分组处理
    grouped = df.groupby('code')
    # 计算技术指标
    ma10 = grouped['close'].transform(lambda x: x.rolling(10).mean())  # 10日均线
    prev_low = grouped['low'].shift(1)  # 前一日最低价
    prev_volume = grouped['volume'].shift(1)  # 前一日成交量
    # 构建筛选条件
    conditions = (
            (df['close'] > prev_low) &  # 多头排列
            (df['close'] > ma10) &  # 10日线上方
            (df['volume'] > prev_volume) &  # 放量
            (df['volume'] < 10 * prev_volume) &  # 成交量未暴增
            (df['close'] > 1)  # 股价>1
    )
    # 获取最新交易日数据
    latest_data = df[df['time'] == yesterday]
    valid_stocks = latest_data[conditions]['code'].unique().tolist()

    return valid_stocks


##国九条筛选##
def GJT_filter_stocks(stocks):
    # 国九更新：过滤近一年净利润为负且营业收入小于1亿的
    # 国九更新：过滤近一年期末净资产为负的 (经查询没有为负数的，所以直接pass这条)
    q = query(
        valuation.code,
        valuation.market_cap,  # 总市值 circulating_market_cap/market_cap
        income.np_parent_company_owners,  # 归属于母公司所有者的净利润
        income.net_profit,  # 净利润
        income.operating_revenue  # 营业收入
        # security_indicator.net_assets
    ).filter(
        valuation.code.in_(stocks),
        income.np_parent_company_owners > 0,
        income.net_profit > 0,
        income.operating_revenue > 1e8,
        indicator.roe > 0,
        indicator.roa > 0,
    )
    df = get_fundamentals(q)

    final_list = list(df.code)

    return final_list
