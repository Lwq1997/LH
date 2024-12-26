# 克隆自聚宽文章：https://www.joinquant.com/post/50813
# 标题：Y神首版低开策略优化--AI解读与提速
# 作者：komunling

from jqlib.technical_analysis import *
from jqfactor import *
from jqdata import *
import datetime as dt
import pandas as pd


def initialize(context):
    # 系统设置
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)
    log.set_level('system', 'error')
    # 每日运行
    run_daily(buy, '09:30')  # 开盘后立即执行买入
    run_daily(sell, '11:28')  # 上午止盈卖出
    run_daily(sell, '14:50')  # 下午止损卖出


# 选股函数
def buy(context):
    # 基础信息
    date = context.previous_date.strftime('%Y-%m-%d')
    current_data = get_current_data()

    # 获取昨日涨停的股票列表
    initial_list = prepare_stock_list(context, date)
    hl_list = get_hl_stock(initial_list, date)

    if hl_list:
        # 获取非连板的股票列表
        ccd = get_continue_count_df(hl_list, date, 10)
        lb_list = ccd.index.tolist()
        stock_list = [s for s in hl_list if s not in lb_list]
        log.info('stock_list:', stock_list)
        # 计算相对位置，筛选相对位置 ≤ 0.5 的股票
        rpd = get_relative_position_df(stock_list, date, 60)
        log.info('rpd:', rpd)
        if not rpd.empty:
            stock_list = rpd[rpd['rp'] <= 0.5].index.tolist()
        else:
            stock_list = []

        # 筛选今日低开的股票（开盘价在昨日收盘价的96%至97%之间）
        if stock_list:
            # 一次性获取昨日收盘价
            df_close = get_price(stock_list, end_date=date, frequency='daily', fields=['close'], count=1, panel=False)
            df_close = df_close.set_index('code')
            # 获取今日开盘价
            open_prices = {s: current_data[s].day_open for s in stock_list}
            df_close['open'] = pd.Series(open_prices)
            df_close['open_pct'] = df_close['open'] / df_close['close']
            log.info('df_close:', df_close)
            df_close = df_close[(df_close['open_pct'] >= 0.96) & (df_close['open_pct'] <= 0.97)]
            stock_list = df_close.index.tolist()
            log.info('stock_list:', stock_list)

        # 买入操作
        if not context.portfolio.positions and stock_list:
            cash_per_stock = context.portfolio.total_value / len(stock_list)
            for s in stock_list:
                order_target_value(s, cash_per_stock)
                print('买入', [get_security_info(s).display_name, s])
                print('———————————————————————————————————')


# 卖出函数
def sell(context):
    # 基础信息
    current_time = context.current_dt.strftime('%H:%M:%S')
    current_data = get_current_data()

    # 判断当前时间，执行对应的卖出策略
    if current_time == '11:28:00':
        # 止盈卖出
        for s in list(context.portfolio.positions.keys()):
            position = context.portfolio.positions[s]
            if position.closeable_amount > 0:
                last_price = current_data[s].last_price
                high_limit = current_data[s].high_limit
                avg_cost = position.avg_cost
                if last_price < high_limit and last_price > avg_cost:
                    order_target_value(s, 0)
                    print('止盈卖出', [get_security_info(s).display_name, s])
                    print('———————————————————————————————————')

    elif current_time == '14:50:00':
        # 止损卖出
        for s in list(context.portfolio.positions.keys()):
            position = context.portfolio.positions[s]
            if position.closeable_amount > 0:
                last_price = current_data[s].last_price
                high_limit = current_data[s].high_limit
                if last_price < high_limit:
                    order_target_value(s, 0)
                    print('止损卖出', [get_security_info(s).display_name, s])
                    print('———————————————————————————————————')


# 辅助函数

# 获取初始股票池
def prepare_stock_list(context, date):
    initial_list = list(get_all_securities(types=['stock'], date=date).index)
    # 过滤科创板、北交所股票
    initial_list = [stock for stock in initial_list if
                    not (stock.startswith('688') or stock.startswith('4') or stock.startswith('8'))]
    # 过滤次新股（上市未满250天）
    d_date = context.previous_date
    initial_list = [stock for stock in initial_list if (d_date - get_security_info(stock).start_date).days > 250]
    # 获取ST股票列表
    st_info = get_extras('is_st', initial_list, start_date=date, end_date=date, df=True).iloc[0]
    initial_list = st_info[~st_info].index.tolist()
    # 获取停牌股票列表
    paused_info = get_current_data()
    initial_list = [stock for stock in initial_list if not paused_info[stock].paused]
    return initial_list


# 获取昨日涨停的股票
def get_hl_stock(initial_list, date):
    df = get_price(initial_list, end_date=date, frequency='daily', fields=['close', 'high_limit'], count=1, panel=False)
    df = df.dropna()
    hl_list = df[df['close'] == df['high_limit']]['code'].tolist()
    return hl_list


# 获取连续涨停的股票
def get_continue_count_df(hl_list, date, watch_days):
    # 获取最近 watch_days 天的数据
    df = get_price(hl_list, end_date=date, frequency='daily', fields=['close', 'high_limit'], count=watch_days,
                   panel=False)
    df = df.dropna()
    df['is_limit_up'] = df['close'] == df['high_limit']

    # 按股票分组，计算连续涨停天数
    def calc_continue_count(group):
        group = group.sort_values('time', ascending=False)
        count = 0
        for is_limit_up in group['is_limit_up']:
            if is_limit_up:
                count += 1
            else:
                break
        return count

    counts = df.groupby('code').apply(calc_continue_count)
    ccd = counts[counts >= 2]
    return ccd


# 计算相对位置
def get_relative_position_df(stock_list, date, watch_days):
    if not stock_list:
        return pd.DataFrame(columns=['rp'])
    df = get_price(stock_list, end_date=date, frequency='daily', fields=['high', 'low', 'close'], count=watch_days,
                   panel=False)
    df = df.dropna()
    grouped = df.groupby('code')
    high = grouped['high'].max()
    low = grouped['low'].min()
    close = grouped.apply(lambda x: x.iloc[-1]['close'])
    result = pd.DataFrame()
    result['rp'] = (close - low) / (high - low)
    return result
