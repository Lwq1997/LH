# 克隆自聚宽文章：https://www.joinquant.com/post/54504
# 标题：一进二策略，无未来函数
# 作者：慢壹拍

# 克隆自聚宽文章：https://www.joinquant.com/post/52326
# 标题：一进二策略，无未来函数（错误已修复）
# 作者：乐水无畏

# 克隆自聚宽文章：https://www.joinquant.com/post/51962
# 标题：四分之一仓位买入策略
# 作者：jyyyfly

# 链接：https://www.joinquant.com/view/community/detail/0498aee294f4a325b620274485a3e3b5?type=1
"""
如何做好四分之一仓位买入？
聚宽相关API接口：https://www.joinquant.com/help/api/help#name:api

"""

from jqdata import *
from jqfactor import *
from jqlib.technical_analysis import *
import datetime as dt
import pandas as pd
from datetime import datetime
from datetime import timedelta


# 初始化函数
def initialize(context):
    set_option('use_real_price', True)
    log.set_level('system', 'error')
    set_option('avoid_future_data', True)
    # 设置滑点为0，即不考虑滑点对交易的影响
    set_slippage(FixedSlippage(0.01))
    # 设置交易成本
    set_order_cost(OrderCost(open_tax=0, close_tax=0.0005, open_commission=0.0000854, close_commission=0.0000954,
                             close_today_commission=0, min_commission=5), type='stock')
    run_daily(get_stock_list, '08:00:00')  # 每天8:00:00运行选股函数
    run_daily(buy, '09:30:00')  # 每天9:28:01运行买入函数
    run_daily(sell, time='11:28', reference_security='000300.XSHG')  # 每天11:28运行卖出函数，参考股票为沪深300指数
    run_daily(sell, time='14:50', reference_security='000300.XSHG')  # 每天14:50运行卖出函数，参考股票为沪深300指数
    g.unsold_stocks = []  # 初始化一个空列表来存储未卖出的股票


# 选股函数
def get_stock_list(context):
    '''
    # 判断大盘是否在五日均线之上
    if not is_market_above_ma5(context):
        log.info("大盘在五日均线以下，不进行选股操作")
        g.target_list = []
        g.priority_list = []
        g.all_list = []
        return
    '''
    # 初始列表
    initial_list = prepare_stock_list(context)
    # 获取买入目标股票
    g.target_list = get_target_list(context, initial_list)
    # 第二段代码的输出作为优先买入股票
    g.priority_list = get_priority_list(context, initial_list)
    g.all_list = list(dict.fromkeys(g.priority_list + g.target_list))


# 第二段代码的输出作为优先买入股票
def get_target_list(context, initial_list):
    # 获取前一个交易日的日期
    date = transform_date(context.previous_date, 'str')

    # 获取最近两个交易日的日期
    date_1 = get_shifted_date(date, -1, 'T')  # 前一个交易日
    date_2 = get_shifted_date(date, -2, 'T')  # 前前一个交易日

    # 获取这两个交易日之前的两个交易日的日期
    date_3 = get_shifted_date(date, -3, 'T')  # 前前前一个交易日
    date_4 = get_shifted_date(date, -4, 'T')  # 前前前前一个交易日

    # 筛选在最近两个交易日（date 和 date_1）均有涨停的股票
    hl_list_day1 = get_hl_stock(initial_list, date)  # 最近一个交易日涨停
    hl_list_day2 = get_hl_stock(initial_list, date_1)  # 前一个交易日涨停
    hl_list = list(set(hl_list_day1) & set(hl_list_day2))  # 取交集，确保两个交易日均涨停

    # 筛选在更早的两个交易日（date_2 和 date_3）未涨停的股票
    hl_earlier_day1 = get_ever_hl_stock(initial_list, date_2)  # 前前一个交易日涨停
    hl_earlier_day2 = get_ever_hl_stock(initial_list, date_3)  # 前前前一个交易日涨停
    elements_to_remove = set(hl_earlier_day1 + hl_earlier_day2)
    print(date_2,'  ',elements_to_remove)

    # 从hl_list中剔除在date_2和date_3也涨停过的股票
    hl_list = [stock for stock in hl_list if stock not in elements_to_remove]

    buy_str = ''
    qualified_stocks = []

    # 遍历最近一个交易日的涨停股票列表
    for s in hl_list:

        # 条件一：均价，金额，市值，换手率
        prev_day_data = attribute_history(s, 1, '1d', fields=['close', 'volume', 'money'], skip_paused=True)
        avg_price_increase_value = prev_day_data['money'][0] / prev_day_data['volume'][0] / prev_day_data['close'][
            0] * 1.1 - 1
        # 如果平均价格涨幅小于0.07或者前一个交易日的成交金额小于7亿或者大于20亿，则跳过
        if avg_price_increase_value < 0.07 or prev_day_data['money'][0] < 7e8 or prev_day_data['money'][0] > 30e8:
            continue

        # 条件二：换手率
        turnover_ratio_data = get_valuation(s, start_date=context.previous_date, end_date=context.previous_date,
                                            fields=['turnover_ratio', 'market_cap', 'circulating_market_cap'])
        if turnover_ratio_data.empty or turnover_ratio_data['market_cap'][0] < 70 or \
                turnover_ratio_data['circulating_market_cap'][0] > 300:
            continue

        yesterday_turnover_ratio = turnover_ratio_data['turnover_ratio'][0]
        if yesterday_turnover_ratio < 10 or yesterday_turnover_ratio > 30:
            continue

        # 条件三：昨日涨停的成交量为近100日的最大成交量
        # 获取昨日成交量
        yesterday_volume = prev_day_data['volume'][0]
        log.error(f'{s}昨日成交量{yesterday_volume}')
        # 获取过去100个交易日的成交量
        past_volume_data = attribute_history(s, 100, '1d', fields=['volume'], skip_paused=True)
        if past_volume_data.empty:
            continue
        max_past_volume = past_volume_data['volume'].max()
        log.error(f'{s}最大成交量{max_past_volume}')
        if yesterday_volume < max_past_volume:
            continue

        # 条件四： 昨日收盘时封单金额需大于流通市值的2%
        # 获取昨日收盘时的封单金额
        # 使用 get_ticks 获取昨日最后一笔的盘口数据

        edate = context.previous_date
        end_time = str(edate) + ' ' + '15:00:00'
        ticks = get_ticks(s, end_dt=end_time, count=1, fields=['time', 'a1_v', 'a1_p', 'b1_v', 'b1_p'], skip=False,
                          df=True)
        if len(ticks) == 0:
            continue

        bid_volume = ticks['b1_p'].iloc[0]
        bid_price = ticks['b1_v'].iloc[0]
        # 计算封单金额
        order_amount = bid_volume * bid_price
        # 获取流通市值
        circulating_market_cap = turnover_ratio_data['circulating_market_cap'][0]
        # 计算封单金额占流通市值的比例
        order_ratio = order_amount / (circulating_market_cap * 100000000)
        if order_ratio < 0.01:
            continue

        df = get_price(s, end_date=date, frequency='daily', fields=['low', 'close', 'low_limit'], count=10, panel=False,
                       fill_paused=False, skip_paused=False)
        low_limit_count = len(df[df.close == df.low_limit])
        if low_limit_count >= 1:
            continue
        # 将符合条件的股票添加到保存的股票列表中
        qualified_stocks.append(s)
        buy_str += "%s%s;" % (s, get_security_info(s).display_name)

    # send_message('可能买入股票：%s '%(buy_str))
    log.info('可能买入target一股票: %s' % buy_str)
    return qualified_stocks


# 第二段代码的输出作为优先买入股票
def get_priority_list(context, initial_list):
    # 将日期转换为字符串格式
    date = transform_date(context.previous_date, 'str')
    # 获取前一个交易日的日期
    date_1 = get_shifted_date(date, -1, 'T')
    # 获取前两个交易日的日期
    date_2 = get_shifted_date(date, -2, 'T')
    # 获取前两个交易日的日期
    # date_3 = get_shifted_date(date, -3, 'T')
    # date_4 = get_shifted_date(date, -4, 'T')
    # 获取最近一个交易日的涨停股票列表
    hl_list = get_hl_stock(initial_list, date)
    # 获取最近两个交易日的涨停股票列表
    hl1_list = get_ever_hl_stock(initial_list, date_1)
    # 获取最近三个交易日的涨停股票列表
    hl2_list = get_ever_hl_stock(initial_list, date_2)
    # hl3_list = get_ever_hl_stock(initial_list, date_3)
    # hl4_list = get_ever_hl_stock(initial_list, date_4)
    # 将最近两个交易日的涨停股票列表和最近三个交易日的涨停股票列表合并
    elements_to_remove = set(hl1_list + hl2_list)
    # 从最近一个交易日的涨停股票列表中移除最近两个交易日的涨停股票列表和最近三个交易日的涨停股票列表中的股票
    hl_list = [stock for stock in hl_list if stock not in elements_to_remove]

    buy_str = ''
    qualified_stocks = []

    # 遍历最近一个交易日的涨停股票列表
    for s in hl_list:

        # 过滤前面三天涨幅超过18%的票
        price_data = attribute_history(s, 4, '1d', fields=['close'], skip_paused=True)
        if len(price_data) < 4:
            continue
        increase_ratio = (price_data['close'][-1] - price_data['close'][0]) / price_data['close'][0]
        if increase_ratio > 0.23:
            continue
        # 条件一：均价，金额，市值，换手率
        prev_day_data = attribute_history(s, 1, '1d', fields=['close', 'volume', 'money'], skip_paused=True)
        avg_price_increase_value = prev_day_data['money'][0] / prev_day_data['volume'][0] / prev_day_data['close'][
            0] * 1.1 - 1
        # 如果平均价格涨幅小于0.07或者前一个交易日的成交金额小于7亿或者大于20亿，则跳过
        if avg_price_increase_value < 0.07 or prev_day_data['money'][0] < 7e8 or prev_day_data['money'][0] > 20e8:
            continue
        # 如果换手率为空或者市值小于70，则跳过
        turnover_ratio_data = get_valuation(s, start_date=context.previous_date, end_date=context.previous_date,
                                            fields=['turnover_ratio', 'market_cap', 'circulating_market_cap'])
        if turnover_ratio_data.empty or turnover_ratio_data['market_cap'][0] < 70 or \
                turnover_ratio_data['circulating_market_cap'][0] > 300:
            continue
        # 如果近期有跌停，则跳过
        df = get_price(s, end_date=date, frequency='daily', fields=['low', 'close', 'low_limit'], count=10, panel=False,
                       fill_paused=False, skip_paused=False)
        low_limit_count = len(df[df.close == df.low_limit])
        if low_limit_count >= 1:
            continue

        # 条件二：左压

        zyts = calculate_zyts(s, context)
        volume_data = attribute_history(s, zyts, '1d', fields=['volume'], skip_paused=True)
        if len(volume_data) < 2 or volume_data['volume'][-1] <= max(volume_data['volume'][:-1]) * 0.90:
            continue

        # 将符合条件的股票添加到保存的股票列表中
        qualified_stocks.append(s)
        buy_str += "%s%s;" % (s, get_security_info(s).display_name)

    # send_message('可能买入股票：%s '%(buy_str))
    log.info('可能买入target二股票: %s' % buy_str)
    return qualified_stocks


# 交易函数
def buy(context):
    if g.unsold_stocks:  # 如果有未卖出的股票
        print("有未卖出的股票，不允许买入")
        return  # 直接返回，不允许买入

    qualified_stocks = []
    current_data = get_current_data()
    date_now = context.current_dt.strftime("%Y-%m-%d")
    start = date_now + '09:15:00'
    end = date_now + '09:28:00'

    # 先购买第二段代码选出的票
    for s in g.all_list:

        # 集合竞价成交量筛选
        volume_data = attribute_history(s, 1, '1d', fields=['volume'], skip_paused=True)
        auction_data = get_call_auction(s, start_date=start, end_date=end, fields=['time', 'volume', 'current'])
        auction_volume = auction_data['volume'][0] / volume_data['volume'][-1]
        print(s, get_security_info(s).display_name, auction_volume)
        if auction_data.empty or auction_volume < 0.025:
            continue

        # 集合竞价的价格筛选
        current_ratio = auction_data['current'][0] / (current_data[s].high_limit / 1.1)
        print(s, get_security_info(s).display_name, current_ratio)
        if current_ratio <= 0.98 or current_ratio >= 1.07:
            continue

        '''
        # 获取股票流通股本
        turnover_ratio_data = get_valuation(s, start_date=context.previous_date, end_date=context.previous_date, fields=['turnover_ratio','market_cap','circulating_market_cap'])

        # 计算集合竞价换手率
        turnover_rate = (auction_data['volume'][0] / turnover_ratio_data['circulating_market_cap'][0]) 
        print(s, get_security_info(s).display_name,turnover_rate)
        if turnover_rate < 0.005:
            continue
        '''
        qualified_stocks.append(s)

    # 如果第二段代码的输出不为空，则直接购买，不再考虑第一段代码的输出
    if len(qualified_stocks) > 0:
        value = context.portfolio.available_cash / len(qualified_stocks)
        for s in qualified_stocks:
            if context.portfolio.available_cash / current_data[s].last_price > 100:
                order_value(s, value)
                g.unsold_stocks.append(s)  # 添加到未卖出列表
                print('优先买入股票：%s%s' % (s, get_security_info(s).display_name))
                print('———————————————————————————————————')
        return


# 处理日期相关函数
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


# 每日初始股票池
def prepare_stock_list(context):
    initial_list = get_all_securities('stock', context.previous_date.strftime('%Y-%m-%d')).index.tolist()
    initial_list = filter_basic_stock(context, initial_list)
    return initial_list


# 基础过滤(过滤科创北交、ST、停牌、次新股)
def filter_basic_stock(context, stock_list):
    current_data = get_current_data()
    return [stock for stock in stock_list
            if not current_data[stock].paused
            and not current_data[stock].is_st
            and "ST" not in current_data[stock].name
            and "*" not in current_data[stock].name
            and "退" not in current_data[stock].name
            # 过滤从主板退到三板的股票(4开头)，北交所(8开头)， 科创板(688开头)，创业板(300开头)
            and not (stock[0] == "4" or stock[0] == "8" or stock[:2] == "68" or stock[0] == "3")
            # 过滤上市时间少于375个交易日的股票
            and not context.previous_date - get_security_info(stock).start_date < dt.timedelta(375)
            ]


# 计算左压天数
def calculate_zyts(s, context):
    high_prices = attribute_history(s, 101, '1d', fields=['high'], skip_paused=True)['high']
    prev_high = high_prices.iloc[-1]
    zyts_0 = next((i - 1 for i, high in enumerate(high_prices[-3::-1], 2) if high >= prev_high), 100)
    zyts = zyts_0 + 5
    return zyts


# 筛选出某一日涨停的股票
def get_hl_stock(initial_list, date):
    df = get_price(initial_list, end_date=date, frequency='daily', fields=['close', 'high_limit'], count=1, panel=False,
                   fill_paused=False, skip_paused=False)
    df = df.dropna()  # 去除停牌
    df = df[df['close'] == df['high_limit']]
    hl_list = list(df.code)
    return hl_list


# 筛选曾涨停
def get_ever_hl_stock(initial_list, date):
    df = get_price(initial_list, end_date=date, frequency='daily', fields=['close', 'high', 'high_limit'], count=1,
                   panel=False, fill_paused=False, skip_paused=False)
    df = df.dropna()  # 去除停牌
    cd1 = df['high'] == df['high_limit']
    cd2 = df['close'] != df['high_limit']
    df = df[cd1 & cd2]  # 满足涨停且不是涨停收盘
    hl_list = list(df.code)
    return hl_list


# 计算涨停数
def get_hl_count_df(hl_list, date, watch_days):
    # 获取watch_days的数据
    df = get_price(hl_list, end_date=date, frequency='daily', fields=['close', 'high_limit', 'low'], count=watch_days,
                   panel=False, fill_paused=False, skip_paused=False)
    df.index = df.code
    # 计算涨停与一字涨停数，一字涨停定义为最低价等于涨停价
    hl_count_list = []
    extreme_hl_count_list = []
    for stock in hl_list:
        df_sub = df.loc[stock]
        hl_days = df_sub[df_sub.close == df_sub.high_limit].high_limit.count()
        extreme_hl_days = df_sub[df_sub.low == df_sub.high_limit].high_limit.count()
        hl_count_list.append(hl_days)
        extreme_hl_count_list.append(extreme_hl_days)
    # 创建df记录
    df = pd.DataFrame(index=hl_list, data={'count': hl_count_list, 'extreme_count': extreme_hl_count_list})
    return df


# 计算昨涨幅
def get_index_increase_ratio(index_code, context):
    # 获取指数昨天和前天的收盘价
    close_prices = attribute_history(index_code, 2, '1d', fields=['close'], skip_paused=True)
    if len(close_prices) < 2:
        return 0  # 如果数据不足，返回0
    day_before_yesterday_close = close_prices['close'][0]
    yesterday_close = close_prices['close'][1]

    # 计算涨幅
    increase_ratio = (yesterday_close - day_before_yesterday_close) / day_before_yesterday_close
    return increase_ratio


# 计算相对位置
def get_relative_position_df(hl_list, date, watch_days):
    if len(hl_list) != 0:
        df = get_price(hl_list, end_date=date, fields=['high', 'low', 'close'], count=watch_days, fill_paused=False,
                       skip_paused=False, panel=False).dropna()
        close = df.groupby('code').apply(lambda df: df.iloc[-1, -1])
        high = df.groupby('code').apply(lambda df: df['high'].max())
        low = df.groupby('code').apply(lambda df: df['low'].min())
        result = pd.DataFrame()
        result['rp'] = (close - low) / (high - low)
        return result
    else:
        return pd.DataFrame(columns=['rp'])


# 判断大盘是否在五日均线之上
def is_market_above_ma5(context):
    market_code = '000300.XSHG'
    prices = attribute_history(market_code, 5, '1d', fields=['close'], skip_paused=True)
    if len(prices) < 5:
        return False
    ma5 = prices['close'].mean()
    current_close = prices['close'][-1]
    return current_close > ma5


# 上午有利润就跑
def sell(context):
    # 基础信息
    date = transform_date(context.previous_date, 'str')
    current_data = get_current_data()

    # 根据时间执行不同的卖出策略
    if str(context.current_dt)[-8:-3] == '11:28':
        for s in list(context.portfolio.positions):
            if ((context.portfolio.positions[s].closeable_amount != 0) and (
                    current_data[s].last_price < current_data[s].high_limit)):
                order_target_value(s, 0)
                if s in g.unsold_stocks:  # 检查是否存在于未卖出列表中
                    g.unsold_stocks.remove(s)  # 从未卖出列表中移除
                print('止盈卖出股票：%s%s' % (s, get_security_info(s, date).display_name))
                print('———————————————————————————————————')

    if str(context.current_dt)[-8:-3] == '14:50':
        for s in list(context.portfolio.positions):
            if ((context.portfolio.positions[s].closeable_amount != 0) and (
                    current_data[s].last_price < current_data[s].high_limit)):  # avg_cost当前持仓成本
                order_target_value(s, 0)
                if s in g.unsold_stocks:  # 检查是否存在于未卖出列表中
                    g.unsold_stocks.remove(s)  # 从未卖出列表中移除
                print('止损卖出股票：%s%s' % (s, get_security_info(s, date).display_name))
                print('———————————————————————————————————')

    g.unsold_stocks = [s for s in context.portfolio.positions if context.portfolio.positions[s].closeable_amount > 0]




