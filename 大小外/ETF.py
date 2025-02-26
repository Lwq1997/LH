# 克隆自聚宽文章：https://www.joinquant.com/post/51870
# 标题：大小+ETF轮动魔改版本，最大回撤控制在9%左右
# 作者：Tkiiil

from jqdata import *
from jqfactor import *
import numpy as np
import pandas as pd
import pickle
import talib
import warnings

import hashlib
import hmac
import requests
import base64
import time
from jqlib.technical_analysis import *

warnings.filterwarnings("ignore")


# 初始化函数
def initialize(context):
    # 设定基准
    set_benchmark('000300.XSHG')
    # 用真实价格交易
    set_option('use_real_price', True)
    # 打开防未来函数
    set_option("avoid_future_data", True)
    # 将滑点设置为0
    set_slippage(FixedSlippage(0))
    # 设置交易成本万分之三，不同滑点影响可在归因分析中查看
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001, open_commission=0.0003, close_commission=0.0003,
                             close_today_commission=0, min_commission=5), type='stock')
    # 过滤order中低于error级别的日志
    log.set_level('order', 'error')
    # 初始化全局变量
    g.no_trading_today_signal = False
    g.market_temperature = "warm"
    g.stock_num = 5
    g.highest = 50
    g.buy_stock_count = 5
    g.hold_list = []  # 当前持仓的全部股票
    g.yesterday_HL_list = []  # 记录持仓中昨日涨停的股票
    g.stop_loss_pct = 0.06  # 止损线 8%
    g.stop_profit_pct = 0.05  # 止盈线，最大回撤10%
    g.max_price = {}  # 记录每只股票的最高价
    g.singal = 'etf'  # 初始化市场风格信号为etf
    # ETF池配置
    g.etf_pool = {
        'foreign': [
            '513100.XSHG',  # 纳指ETF
            '513500.XSHG',  # 标普500
            '159941.XSHE',  # 纳指100
        ],
        'bond': [
            '511010.XSHG',  # 国债ETF
            '511260.XSHG',  # 十年国债
        ],
        'style': [
            '515180.XSHG',  # 红利ETF
            '563020.XSHG',  # 低波动红利
            '512100.XSHG',  # 中证1000
            # '159928.XSHE',  # 消费ETF
            # '512690.XSHG',  # 酒ETF
        ]
    }
    g.etf_check_period = 20  # ETF筛选周期（交易日）
    g.last_etf_check = None  # 上次ETF筛选时间

    # 设置交易运行时间
    run_daily(prepare_stock_list, '9:25')  # 盘前准备股票列表
    run_daily(check_market_style, '9:30')  # 每日检查市场风格
    run_daily(stop_loss, '10:30')  # 早盘止损止盈
    run_daily(stop_loss, '13:50')  # 午盘止损止盈

    # 市场风格判断参数
    g.style_check_period = 10  # 市场风格判断周期（交易日）
    g.width_check_period = 5  # 市场宽度计算周期（固定为5天）
    g.singal = 'etf'  # 初始化市场风格信号为etf

    # 信号一致性检查参数
    g.signal_queue_size = 3  # 信号队列长度（降低到3天）
    g.signal_consistency = 0.67  # 信号一致性阈值（2/3）
    g.signal_queue = []  # 信号队列

    # 紧急情况阈值
    g.emergency_threshold = {
        'single_day': -0.03,  # 单日跌幅阈值（3%）
        'five_day': -0.06  # 五日跌幅阈值（6%）
    }

    # 调整期货相关参数
    g.futures_code = 'IF'  # 股指期货主力合约
    g.futures_check_period = 5  # 期货市场情绪检查周期
    g.futures_thresholds = {
        'basis': 0.001,  # 基差阈值降低到0.1%
        'position_change': 0.05,  # 持仓变化阈值降低到5%
        'volume_ratio': 1.2  # 成交量放大阈值降低到1.2倍
    }


def getStockIndustry(p_stocks, p_industries_type, p_day):
    # type: (list, str, datetime.date) -> pd.Series
    """
    返回股票代码与所属行业汉字名称的对照表
    :param p_stocks: 股票代码列表
    :param p_industries_type: 行业分类标准，例如sw_l1
    :param p_day: 日期
    :return: pd.Series
    """
    dict_stk_2_ind = {}
    stocks_industry_dict = get_industry(p_stocks, date=p_day)
    for stock in stocks_industry_dict:
        if p_industries_type in stocks_industry_dict[stock]:
            dict_stk_2_ind[stock] = stocks_industry_dict[stock][p_industries_type]['industry_code']
    #
    return pd.Series(dict_stk_2_ind)


def get_industry_width(p_end_date, p_count, p_industries_type):
    # 行业代码，行业名称
    s_industry = get_industries(name=p_industries_type, date=p_end_date)['name']
    s_industry.loc['999998'] = '全市场'
    s_industry.loc['999999'] = '合计'
    #
    trade_days = get_trade_days(end_date=p_end_date, count=p_count + 20)
    stock_list = list(get_all_securities(date=trade_days[0]).index)  # 最早的day之前20天就已经上市的股票
    s_stk_2_ind = getStockIndustry(p_stocks=stock_list, p_industries_type=p_industries_type, p_day=p_end_date)

    # 取数
    h = get_price(stock_list, end_date=p_end_date, frequency='1d', fields=['close'], count=p_count + 20, panel=False)
    h['date'] = pd.DatetimeIndex(h.time).date
    df_close = h.pivot(index='code', columns='date', values='close').dropna(axis=0)
    df_ma20 = df_close.rolling(window=20, axis=1).mean().iloc[:, -p_count:]

    df_bias = (df_close.iloc[:, -p_count:] > df_ma20)  # type: pd.DataFrame
    # 每个交易日全市场的总体状况：Close在MA20之上的比例
    s_mkt_ratio = ((100.0 * df_bias.sum()) / df_bias.count()).round()
    df_bias['industry_code'] = s_stk_2_ind

    # df_ratio: index: 行业代码, columns: 日期
    df_ratio = ((df_bias.groupby('industry_code').sum() * 100.0) / df_bias.groupby(
        'industry_code').count()).round()  # type: pd.DataFrame
    #
    s_mkt_sum = df_ratio.sum()  # 每日合计
    #
    df_ratio.loc['999998'] = s_mkt_ratio
    df_ratio.loc['999999'] = s_mkt_sum
    # 行业汉字名称
    df_ratio['name'] = s_industry
    #
    df_result = df_ratio.set_index('name').T
    #
    for col in df_result.columns:
        df_result[col] = df_result[col].astype("int32")
    #
    df_result.sort_index(ascending=False, inplace=True)
    df_result.index.name = ''
    df_result.columns.name = ''
    #
    # print(df_result)
    return df_result


def prepare_stock_list(context):
    print('每日运行已开启')
    # 使用列表推导式优化获取持仓列表
    g.hold_list = [position.security for position in list(context.portfolio.positions.values())]

    # 优化清理最高价记录
    stocks_to_remove = set(g.max_price.keys()) - set(g.hold_list)
    for stock in stocks_to_remove:
        del g.max_price[stock]

    # 获取昨日涨停列表
    if g.hold_list:
        df = get_price(g.hold_list, end_date=context.previous_date, frequency='daily',
                       fields=['close', 'high_limit'], count=1, panel=False, fill_paused=False)
        g.yesterday_HL_list = list(df[df['close'] == df['high_limit']]['code'])
    else:
        g.yesterday_HL_list = []


def stop_loss(context):
    if not g.hold_list:  # 如果没有持仓，直接返回
        return

    now_time = context.current_dt

    # 处理昨日涨停股票
    if g.yesterday_HL_list:
        # 批量获取价格数据
        current_data = get_price(g.yesterday_HL_list, end_date=now_time, frequency='1m',
                                 fields=['close', 'high_limit'], skip_paused=False,
                                 fq='pre', count=1, panel=False, fill_paused=True)

        for stock in g.yesterday_HL_list:
            stock_data = current_data[current_data['code'] == stock].iloc[0]
            if stock_data['close'] < stock_data['high_limit']:
                log.info("[%s]涨停打开，卖出" % (stock))
                order_target_value(stock, 0)
                log.debug("止损 Selling out %s" % (stock))
            else:
                log.info("[%s]涨停，继续持有" % (stock))

    # 批量获取当前持仓的价格数据
    positions = context.portfolio.positions
    position_stocks = list(positions.keys())
    if not position_stocks:  # 如果没有持仓，直接返回
        return

    # 执行止损和止盈
    for stock in position_stocks:
        position = positions[stock]
        current_price = position.price
        cost = position.avg_cost

        # 更新最高价记录
        if stock not in g.max_price:
            g.max_price[stock] = current_price
        elif current_price > g.max_price[stock]:
            g.max_price[stock] = current_price

        # 止损检查
        if current_price < cost * (1 - g.stop_loss_pct):
            order_target_value(stock, 0)
            log.debug("止损卖出 %s, 成本: %.2f, 现价: %.2f, 跌幅: %.2f%%" %
                      (stock, cost, current_price, (current_price / cost - 1) * 100))
            del g.max_price[stock]

        # 止盈检查
        elif current_price < g.max_price[stock] * (1 - g.stop_profit_pct):
            order_target_value(stock, 0)
            log.debug("止盈卖出 %s, 最高价: %.2f, 现价: %.2f, 回撤: %.2f%%" %
                      (stock, g.max_price[stock], current_price,
                       (current_price / g.max_price[stock] - 1) * 100))
            del g.max_price[stock]


def filter_roic(context, stock_list):
    yesterday = context.previous_date
    # 批量获取ROIC数据
    roic_data = get_factor_values(stock_list, 'roic_ttm', end_date=yesterday, count=1)['roic_ttm']
    # 使用更严格的过滤条件
    valid_stocks = []
    for stock in stock_list:
        try:
            roic = roic_data[stock].iloc[0]
            if roic is not None and roic > 0.08:
                valid_stocks.append(stock)
        except:
            continue
    return valid_stocks


def filter_highprice_stock(context, stock_list):
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
            or last_prices[stock][-1] < 10]


def filter_highprice_stock2(context, stock_list):
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
            or last_prices[stock][-1] < 300]


def get_recent_limit_up_stock(context, stock_list, recent_days):
    if not stock_list:  # 如果列表为空直接返回
        return []
    stat_date = context.previous_date
    # 批量获取数据
    df = get_price(stock_list, end_date=stat_date, frequency='daily',
                   fields=['close', 'high_limit'], count=recent_days,
                   panel=False, fill_paused=False)
    # 使用groupby和transform优化判断逻辑
    limit_up_mask = (df['close'] == df['high_limit'])
    has_limit_up = limit_up_mask.groupby(df['code']).any()
    return list(has_limit_up[has_limit_up].index)


def get_recent_down_up_stock(context, stock_list, recent_days):
    if not stock_list:  # 如果列表为空直接返回
        return []
    stat_date = context.previous_date
    # 批量获取数据
    df = get_price(stock_list, end_date=stat_date, frequency='daily',
                   fields=['close', 'low_limit'], count=recent_days,
                   panel=False, fill_paused=False)
    # 使用groupby和transform优化判断逻辑
    limit_down_mask = (df['close'] == df['low_limit'])
    has_limit_down = limit_down_mask.groupby(df['code']).any()
    return list(has_limit_down[has_limit_down].index)


# 1-2 选股模块
def get_stock_list(context):
    final_list = []
    MKT_index = '399101.XSHE'
    initial_list = get_index_stocks(MKT_index)
    initial_list = filter_new_stock(context, initial_list)
    initial_list = filter_kcbj_stock(initial_list)
    initial_list = filter_st_stock(initial_list)

    q = query(valuation.code, valuation.market_cap).filter(valuation.code.in_(initial_list),
                                                           valuation.market_cap.between(5, 30)).order_by(
        valuation.market_cap.asc())
    df_fun = get_fundamentals(q)
    df_fun = df_fun[:100]

    initial_list = list(df_fun.code)
    initial_list = filter_paused_stock(initial_list)
    initial_list = filter_limitup_stock(context, initial_list)
    initial_list = filter_limitdown_stock(context, initial_list)
    # print('initial_list中含有{}个元素'.format(len(initial_list)))
    q = query(valuation.code, valuation.market_cap).filter(valuation.code.in_(initial_list)).order_by(
        valuation.market_cap.asc())
    df_fun = get_fundamentals(q)
    df_fun = df_fun[:50]
    final_list = list(df_fun.code)
    return final_list


# 1-2 选股模块
def get_stock_list_2(context):
    final_list = []
    MKT_index = '399101.XSHE'
    initial_list = get_index_stocks(MKT_index)
    initial_list = filter_new_stock(context, initial_list)
    initial_list = filter_kcbj_stock(initial_list)
    initial_list = filter_st_stock(initial_list)
    # 国九更新：过滤近一年净利润为负且营业收入小于1亿的
    # 国九更新：过滤近一年期末净资产为负的 (经查询没有为负数的，所以直接pass这条)
    # 国九更新：过滤近一年审计建议无法出具或者为负面建议的 (经过净利润等筛选，审计意见几乎不会存在异常)
    q = query(
        valuation.code,
        valuation.market_cap,  # 总市值 circulating_market_cap/market_cap
        income.np_parent_company_owners,  # 归属于母公司所有者的净利润
        income.net_profit,  # 净利润
        income.operating_revenue  # 营业收入
        # security_indicator.net_assets
    ).filter(
        valuation.code.in_(initial_list),
        valuation.market_cap.between(5, 30),
        income.np_parent_company_owners > 0,
        income.net_profit > 0,
        income.operating_revenue > 1e8
    ).order_by(valuation.market_cap.asc()).limit(50)

    df = get_fundamentals(q)

    final_list = list(df.code)
    last_prices = history(1, unit='1d', field='close', security_list=final_list)

    return [stock for stock in final_list if stock in g.hold_list or last_prices[stock][-1] <= g.highest]


def SMALL(context, choice):
    """
    优化后的小盘股票选择函数
    1. 扩大选股池
    2. 加强基本面筛选
    3. 增加动量因子
    """
    dt_last = context.previous_date

    # 1. 获取两个指数的成分股
    stocks_zz1000 = set(get_index_stocks('000852.XSHG', dt_last))  # 中证1000
    stocks_zxb = set(get_index_stocks('399101.XSHE', dt_last))  # 中小板

    # 合并股票池并转为列表
    initial_stocks = list(stocks_zz1000.union(stocks_zxb))

    # 基础过滤
    initial_stocks = filter_new_stock(context, initial_stocks)
    initial_stocks = filter_kcbj_stock(initial_stocks)
    initial_stocks = filter_st_stock(initial_stocks)

    # 2. 市值过滤和基本面筛选
    q = query(
        valuation.code,
        valuation.market_cap,
        indicator.roe,
        indicator.roa,
        indicator.inc_revenue_year_on_year,  # 营收增速
        indicator.inc_net_profit_year_on_year,  # 净利润增速
        balance.total_liability,  # 总负债
        balance.total_assets,  # 总资产
        cash_flow.net_operate_cash_flow,  # 经营活动现金流
        income.net_profit  # 净利润
    ).filter(
        valuation.code.in_(initial_stocks),
        valuation.market_cap.between(5, 30),  # 市值范围5-30亿
        indicator.roe > 0.08,  # ROE > 8%
        indicator.roa > 0.05,  # ROA > 5%
        indicator.inc_revenue_year_on_year > 15,  # 营收增速 > 15%
        indicator.inc_net_profit_year_on_year > 15  # 净利润增速 > 15%
    )

    df = get_fundamentals(q, dt_last)

    # 计算资产负债率和现金流利润比
    df['debt_ratio'] = df['total_liability'] / df['total_assets']  # 资产负债率
    df['cash_profit_ratio'] = df['net_operate_cash_flow'] / df['net_profit']  # 现金流利润比

    # 进一步筛选
    df = df[
        (df['debt_ratio'] < 0.6) &  # 资产负债率 < 60%
        (df['cash_profit_ratio'] > 0.8)  # 经营性现金流/净利润 > 0.8
        ]

    if df.empty:
        print("没有满足基本面条件的股票")
        return []

    # 3. 计算动量因子
    stock_list = list(df['code'])

    # 获取历史收盘价数据
    price_df = get_price(stock_list,
                         end_date=dt_last,
                         frequency='daily',
                         fields=['close'],
                         count=120,  # 120天数据
                         panel=False)

    # 计算不同期限的动量
    momentum_scores = {}

    for stock in stock_list:
        try:
            stock_prices = price_df[price_df['code'] == stock]['close'].values

            # 计算不同时间窗口的动量
            m20 = stock_prices[-1] / stock_prices[-20] - 1  # 20日动量
            m60 = stock_prices[-1] / stock_prices[-60] - 1  # 60日动量
            m120 = stock_prices[-1] / stock_prices[-120] - 1  # 120日动量

            # 综合动量得分
            momentum_score = 0.5 * m20 + 0.3 * m60 + 0.2 * m120
            momentum_scores[stock] = momentum_score

        except Exception as e:
            print(f"计算动量得分出错: {stock}, {str(e)}")
            momentum_scores[stock] = float('-inf')

    # 按动量得分排序并选取前5只股票
    sorted_stocks = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
    final_stocks = [stock for stock, score in sorted_stocks[:g.stock_num]]

    print(f"最终选出的股票: {final_stocks}")
    print("动量得分:")
    for stock in final_stocks:
        print(f"{stock}: {momentum_scores[stock]:.2%}")

    return final_stocks


def check_market_style(context):
    """
    每日检查市场风格
    判断顺序：
    1. 紧急情况判断
    2. 期货情绪 + 市场宽度综合判断
    3. 大小盘涨幅判断
    """
    dt_last = context.previous_date
    print(f"\n=== 市场风格判断开始 [{dt_last}] ===")
    print(f"当前信号: {g.singal}")
    print(f"信号队列: {g.signal_queue}")

    # 1. 紧急情况判断
    print("\n--- 1. 紧急情况判断 ---")
    if check_emergency(context):
        old_signal = g.singal
        g.singal = 'etf'
        g.signal_queue = []  # 清空信号队列
        if old_signal != g.singal:
            print(f'触发紧急情况，切换信号: {old_signal} -> {g.singal}')
            monthly_adjustment(context)
        return
    else:
        print("未触发紧急情况")

    # 2. 期货情绪 + 市场宽度综合判断
    print("\n--- 2. 期货情绪和市场宽度判断 ---")
    futures_sentiment = check_futures_sentiment(context)
    width_data = get_industry_width(dt_last, g.width_check_period, 'sw_l1')
    width_change = width_data['全市场'][1] - width_data['全市场'][0]
    print(f"市场宽度数据:")
    print(f"起始值: {width_data['全市场'][0]}")
    print(f"结束值: {width_data['全市场'][1]}")
    print(f"变化值: {width_change}")

    # 3. 计算大小盘涨幅
    print("\n--- 3. 大小盘涨幅计算 ---")
    print("获取大盘股票池...")
    B_stocks = get_index_stocks('000300.XSHG', dt_last)
    B_stocks = filter_kcbj_stock(B_stocks)
    B_stocks = filter_st_stock(B_stocks)
    B_stocks = filter_new_stock(context, B_stocks)
    print(f"大盘股票池数量: {len(B_stocks)}")

    print("\n获取小盘股票池...")
    S_stocks = get_index_stocks('399101.XSHE', dt_last)
    S_stocks = filter_kcbj_stock(S_stocks)
    S_stocks = filter_st_stock(S_stocks)
    S_stocks = filter_new_stock(context, S_stocks)
    print(f"小盘股票池数量: {len(S_stocks)}")

    # 获取大盘前20只股票
    print("\n选取大盘前20只股票...")
    q = query(valuation.code, valuation.circulating_market_cap
              ).filter(valuation.code.in_(B_stocks)
                       ).order_by(valuation.circulating_market_cap.desc())
    df = get_fundamentals(q, date=dt_last)
    Blst = list(df.code)[:20]
    print(f"大盘样本: {Blst}")

    # 获取小盘前20只股票
    print("\n选取小盘前20只股票...")
    q = query(valuation.code, valuation.circulating_market_cap
              ).filter(valuation.code.in_(S_stocks)
                       ).order_by(valuation.circulating_market_cap.asc())
    df = get_fundamentals(q, date=dt_last)
    Slst = list(df.code)[:20]
    print(f"小盘样本: {Slst}")

    # 计算大小盘涨幅（使用配置的周期）
    print(f"\n计算涨幅（周期：{g.style_check_period}天）...")
    B_ratio = get_price(Blst, end_date=dt_last, frequency='1d',
                        fields=['close'], count=g.style_check_period, panel=False
                        ).pivot(index='time', columns='code', values='close')
    change_BIG = (B_ratio.iloc[-1] / B_ratio.iloc[0] - 1) * 100
    B_mean = np.nanmean(change_BIG)
    print(f"大盘个股涨幅: {change_BIG.to_dict()}")
    print(f"大盘平均涨幅: {B_mean:.2f}%")

    S_ratio = get_price(Slst, end_date=dt_last, frequency='1d',
                        fields=['close'], count=g.style_check_period, panel=False
                        ).pivot(index='time', columns='code', values='close')
    change_SMALL = (S_ratio.iloc[-1] / S_ratio.iloc[0] - 1) * 100
    S_mean = np.nanmean(change_SMALL)
    print(f"小盘个股涨幅: {change_SMALL.to_dict()}")
    print(f"小盘平均涨幅: {S_mean:.2f}%")

    # 综合判断
    print("\n--- 4. 综合判断 ---")
    old_signal = g.singal
    new_signal = old_signal

    # 期货情绪强信号判断
    print("\n期货情绪强信号判断:")
    if futures_sentiment == -1 and width_change < -20:
        new_signal = 'etf'
        print(f'期货看空且市场宽度收缩{width_change}，建议切换为ETF')
    elif futures_sentiment == 1 and width_change > 20:
        new_signal = 'big'
        print(f'期货看多且市场宽度扩张{width_change}，建议切换为大盘')
    else:
        print("期货情绪和市场宽度未给出强信号，进入大小盘涨幅判断")
        # 大小盘涨幅判断
        print("\n大小盘涨幅判断:")
        if B_mean > S_mean and B_mean > 0:
            if B_mean > 5:
                new_signal = 'small'
                print(f'大盘涨幅{B_mean:.2f}%超过5%，建议切换为小盘')
            else:
                new_signal = 'big'
                print(f'大盘涨幅{B_mean:.2f}%领先，建议切换为大盘')
        elif B_mean < S_mean and S_mean > 0:
            new_signal = 'small'
            print(f'小盘涨幅{S_mean:.2f}%领先，建议切换为小盘')
        else:
            new_signal = 'etf'
            print(f'大小盘均弱，建议切换为ETF')

    # 信号一致性检查
    print("\n--- 5. 信号一致性检查 ---")
    g.signal_queue.append(new_signal)
    if len(g.signal_queue) > g.signal_queue_size:
        g.signal_queue.pop(0)
    print(f"当前信号队列: {g.signal_queue}")

    # 只有在信号一致性达到要求时才切换
    if check_signal_consistency(new_signal):
        if new_signal != old_signal:
            g.singal = new_signal
            print(f'信号切换: {old_signal} -> {new_signal}')
            monthly_adjustment(context)
    else:
        print(f'信号一致性未达标，保持原信号: {old_signal}')

    # 输出当前市场状况
    print(f"\n=== 市场状况汇总 ===")
    print(f"期货情绪: {['看空', '中性', '看多'][futures_sentiment + 1]}")
    print(f"市场宽度变化: {width_change:.2f}")
    print(f"大盘涨幅: {B_mean:.2f}%")
    print(f"小盘涨幅: {S_mean:.2f}%")
    print(f"当前信号: {g.singal}")
    print(f"信号队列: {g.signal_queue}")
    print("=== 市场风格判断结束 ===\n")


# 添加ETF选择函数
def select_etf(context):
    """
    增强版ETF选择策略
    """
    dt_last = context.previous_date

    # 获取主要指数走势
    indices = {
        'hs300': '000300.XSHG',  # 沪深300
        'nasdaq': '513100.XSHG',  # 纳指ETF
        'bond': '511010.XSHG'  # 国债ETF
    }

    # 计算各指数的走势和波动
    index_metrics = {}
    for name, code in indices.items():
        try:
            data = get_price(code, end_date=dt_last, frequency='daily',
                             fields=['close', 'high', 'low'],
                             count=g.etf_check_period, panel=False)

            # 计算收益率
            returns = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100

            # 计算波动率
            volatility = np.std(data['close'].pct_change()) * np.sqrt(252) * 100

            # 计算最大回撤
            rolling_max = data['close'].rolling(window=g.etf_check_period, min_periods=1).max()
            drawdown = ((data['close'] - rolling_max) / rolling_max).min() * 100

            index_metrics[name] = {
                'returns': returns,
                'volatility': volatility,
                'drawdown': drawdown
            }
        except:
            print(f"获取{name}数据失败")
            continue

    selected_etfs = []

    # 根据市场情况选择ETF
    # 1. 避险配置
    if index_metrics.get('hs300', {}).get('returns', 0) < -5 or \
            index_metrics.get('hs300', {}).get('volatility', 0) > 30:
        # 市场下跌或波动剧烈时，配置债券ETF
        selected_etfs.extend(g.etf_pool['bond'])

    # 2. 海外市场机会
    if index_metrics.get('nasdaq', {}).get('returns', 0) > 3 and \
            index_metrics.get('nasdaq', {}).get('drawdown', -100) > -10:
        # 美股上涨且回撤可控时，配置外盘ETF
        selected_etfs.extend(g.etf_pool['foreign'][:2])

    # 3. 风格ETF筛选
    style_etfs = g.etf_pool['style']
    style_metrics = {}

    for etf in style_etfs:
        try:
            data = get_price(etf, end_date=dt_last, frequency='daily',
                             fields=['close'], count=g.etf_check_period, panel=False)

            returns = (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100
            momentum = (data['close'].iloc[-1] / data['close'].iloc[-5] - 1) * 100

            style_metrics[etf] = returns + momentum  # 综合评分
        except:
            continue

    # 选择评分最高的风格ETF
    sorted_style_etfs = sorted(style_metrics.items(), key=lambda x: x[1], reverse=True)
    selected_etfs.extend([etf for etf, score in sorted_style_etfs[:2] if score > 0])

    # 确保至少持有3只ETF，不超过5只
    if len(selected_etfs) < 3:
        remaining = set(g.etf_pool['bond']) - set(selected_etfs)
        selected_etfs.extend(list(remaining)[:3 - len(selected_etfs)])

    selected_etfs = list(dict.fromkeys(selected_etfs))[:5]  # 去重并限制数量

    print(f"已选择ETF: {selected_etfs}")
    print("选择原因：")
    for etf in selected_etfs:
        if etf in g.etf_pool['bond']:
            print(f"{etf}: 防御配置")
        elif etf in g.etf_pool['foreign']:
            print(f"{etf}: 海外机会")
        else:
            print(f"{etf}: 优势风格")

    return selected_etfs


# 添加紧急情况检查函数
def check_emergency(context):
    """
    检查是否触发紧急情况，增加更多的市场风险指标
    """
    dt_last = context.previous_date

    # 获取沪深300最近5天的数据
    hs300 = get_price('000300.XSHG', end_date=dt_last, frequency='daily',
                      fields=['open', 'high', 'low', 'close', 'volume'],
                      count=5, panel=False)

    # 计算各种风险指标
    # 1. 单日跌幅
    daily_return = (hs300['close'].iloc[-1] / hs300['close'].iloc[-2] - 1)

    # 2. 五日跌幅
    five_day_return = (hs300['close'].iloc[-1] / hs300['close'].iloc[0] - 1)

    # 3. 日内振幅
    daily_amplitude = (hs300['high'].iloc[-1] - hs300['low'].iloc[-1]) / hs300['open'].iloc[-1]

    # 4. 量能变化
    volume_change = (hs300['volume'].iloc[-1] / hs300['volume'].iloc[-2] - 1)

    # 检查是否触发紧急情况
    emergency_triggered = False

    # 单日大跌
    if daily_return < g.emergency_threshold['single_day']:
        print(f"触发单日大跌紧急情况，跌幅：{daily_return:.2%}")
        emergency_triggered = True

    # 五日持续下跌
    if five_day_return < g.emergency_threshold['five_day']:
        print(f"触发五日累计大跌紧急情况，跌幅：{five_day_return:.2%}")
        emergency_triggered = True

    # 日内剧烈波动
    if daily_amplitude > 0.04 and daily_return < -0.02:
        print(f"触发日内剧烈波动，振幅：{daily_amplitude:.2%}，跌幅：{daily_return:.2%}")
        emergency_triggered = True

    # 放量下跌
    if volume_change > 1.5 and daily_return < -0.02:
        print(f"触发放量下跌，量能增幅：{volume_change:.2%}，跌幅：{daily_return:.2%}")
        emergency_triggered = True

    return emergency_triggered


# 添加信号一致性检查函数
def check_signal_consistency(new_signal):
    """
    检查信号一致性
    返回True表示信号一致性达到阈值，可以切换
    """
    # 更新信号队列
    g.signal_queue.append(new_signal)
    if len(g.signal_queue) > g.signal_queue_size:
        g.signal_queue.pop(0)

    # 如果队列未满，允许切换
    if len(g.signal_queue) < g.signal_queue_size:
        return True

    # 计算主导信号的比例
    signal_count = {}
    for signal in g.signal_queue:
        signal_count[signal] = signal_count.get(signal, 0) + 1

    # 获取出现最多的信号及其比例
    dominant_signal = max(signal_count.items(), key=lambda x: x[1])
    consistency = dominant_signal[1] / g.signal_queue_size

    # 检查是否达到一致性阈值
    if consistency >= g.signal_consistency:
        print(f"信号一致性检查通过，主导信号：{dominant_signal[0]}，一致性：{consistency:.2%}")
        return True
    else:
        print(f"信号一致性检查未通过，主导信号：{dominant_signal[0]}，一致性：{consistency:.2%}")
        return False


def check_futures_sentiment(context):
    """
    检查期货市场情绪
    返回值：
    1: 看多
    0: 中性
    -1: 看空
    """
    dt_last = context.previous_date

    try:
        # 获取期货主力合约
        dominant_future = get_future_contracts('IF', dt_last)[0]
        if not dominant_future:
            print("无法获取期货主力合约")
            return 0

        print(f"当前期货合约：{dominant_future}")

        # 获取期货数据（确保数据完整性）
        futures_data = get_price(dominant_future, end_date=dt_last,
                                 frequency='daily',
                                 fields=['open', 'close', 'volume', 'open_interest'],
                                 count=g.futures_check_period,
                                 skip_paused=True)

        # 获取现货数据（沪深300）
        spot_data = get_price('000300.XSHG', end_date=dt_last,
                              frequency='daily',
                              fields=['close'],
                              count=1)

        if futures_data is None or futures_data.empty or spot_data is None or spot_data.empty:
            print("获取数据失败")
            return 0

        print("\n期货数据统计：")
        print(f"期货收盘价：{futures_data['close'].iloc[-1]:.2f}")
        print(f"现货收盘价：{spot_data['close'].iloc[-1]:.2f}")

        # 1. 计算基差（期货-现货）/现货
        basis = (futures_data['close'].iloc[-1] - spot_data['close'].iloc[-1]) / spot_data['close'].iloc[-1]
        print(f"\n1. 基差分析：")
        print(f"基差水平：{basis:.2%}")

        # 2. 计算持仓量变化
        if len(futures_data) >= 2:
            position_change = (futures_data['open_interest'].iloc[-1] / futures_data['open_interest'].iloc[0] - 1)
            print(f"\n2. 持仓量分析：")
            print(f"起始持仓：{futures_data['open_interest'].iloc[0]:.0f}")
            print(f"当前持仓：{futures_data['open_interest'].iloc[-1]:.0f}")
            print(f"持仓变化：{position_change:.2%}")
        else:
            position_change = 0
            print("持仓数据不足")

        # 3. 计算成交量变化和价格趋势
        if len(futures_data) >= 2:
            avg_volume = futures_data['volume'].iloc[:-1].mean()
            volume_ratio = futures_data['volume'].iloc[-1] / avg_volume if avg_volume > 0 else 1

            # 计算价格趋势
            price_trend = (futures_data['close'].iloc[-1] / futures_data['close'].iloc[0] - 1)

            print(f"\n3. 成交量和趋势分析：")
            print(f"平均成交量：{avg_volume:.0f}")
            print(f"当前成交量：{futures_data['volume'].iloc[-1]:.0f}")
            print(f"量比：{volume_ratio:.2f}")
            print(f"价格趋势：{price_trend:.2%}")
        else:
            volume_ratio = 1
            price_trend = 0
            print("成交量数据不足")

        # 计算情绪得分
        sentiment_score = 0
        print("\n情绪打分：")

        # 基差判断（权重：1）
        if basis > g.futures_thresholds['basis']:
            sentiment_score += 1
            print(f"基差升水 {basis:.2%} > {g.futures_thresholds['basis']:.2%}，看多 (+1)")
        elif basis < -g.futures_thresholds['basis']:
            sentiment_score -= 1
            print(f"基差贴水 {basis:.2%} < -{g.futures_thresholds['basis']:.2%}，看空 (-1)")
        else:
            print(f"基差在正常范围内 {basis:.2%}")

        # 持仓量变化判断（权重：1）
        if abs(position_change) > g.futures_thresholds['position_change']:
            if position_change > 0 and price_trend > 0:
                sentiment_score += 1
                print(f"持仓增加 {position_change:.2%} 且价格上涨，看多 (+1)")
            elif position_change > 0 and price_trend < 0:
                sentiment_score -= 1
                print(f"持仓增加 {position_change:.2%} 且价格下跌，看空 (-1)")
        else:
            print(f"持仓变化在正常范围内 {position_change:.2%}")

        # 成交量变化判断（权重：1）
        if volume_ratio > g.futures_thresholds['volume_ratio']:
            if price_trend > 0:
                sentiment_score += 1
                print(f"放量上涨，量比 {volume_ratio:.2f}，看多 (+1)")
            elif price_trend < 0:
                sentiment_score -= 1
                print(f"放量下跌，量比 {volume_ratio:.2f}，看空 (-1)")
        else:
            print(f"成交量在正常范围内，量比 {volume_ratio:.2f}")

        # 价格趋势额外得分（权重：0.5）
        if abs(price_trend) > 0.01:  # 1%的趋势阈值
            if price_trend > 0:
                sentiment_score += 0.5
                print(f"价格趋势向上 {price_trend:.2%}，看多 (+0.5)")
            else:
                sentiment_score -= 0.5
                print(f"价格趋势向下 {price_trend:.2%}，看空 (-0.5)")

        print(f"\n最终情绪得分：{sentiment_score}")

        # 返回最终情绪判断（降低阈值到1.5）
        if sentiment_score >= 1.5:
            print("期货市场情绪：看多")
            return 1
        elif sentiment_score <= -1.5:
            print("期货市场情绪：看空")
            return -1
        else:
            print("期货市场情绪：中性")
            return 0

    except Exception as e:
        print(f"期货情绪检查出错：{str(e)}")
        print(f"错误详情：{traceback.format_exc()}")
        return 0  # 出错时返回中性


# 2-1 过滤停牌股票
def filter_paused_stock(stock_list):
    """
    过滤停牌的股票
    """
    if not stock_list:  # 如果列表为空直接返回
        return []

    current_data = get_current_data()
    filtered_list = []
    for stock in stock_list:
        if not current_data[stock].paused:
            filtered_list.append(stock)
        else:
            print(f"股票{stock}已停牌，被过滤")
    return filtered_list


# 2-2 过滤ST及其他具有退市标签的股票
def filter_st_stock(stock_list):
    """
    过滤ST、*ST、退市等股票
    """
    if not stock_list:  # 如果列表为空直接返回
        return []

    current_data = get_current_data()
    filtered_list = []
    for stock in stock_list:
        if not (current_data[stock].is_st or
                'ST' in current_data[stock].name or
                '*' in current_data[stock].name or
                '退' in current_data[stock].name):
            filtered_list.append(stock)
        else:
            print(f"股票{stock}为ST股票或退市股票，被过滤")
    return filtered_list


# 2-3 过滤科创北交股票
def filter_kcbj_stock(stock_list):
    """
    过滤科创板、创业板和北交所股票
    """
    if not stock_list:  # 如果列表为空直接返回
        return []

    filtered_list = []
    for stock in stock_list:
        # 排除科创板（688开头）、创业板（300开头）、北交所（8开头、4开头）
        if not (stock.startswith('688') or stock.startswith('300') or
                stock.startswith('8') or stock.startswith('4')):
            filtered_list.append(stock)
        else:
            print(f"股票{stock}为科创板/创业板/北交所股票，被过滤")
    return filtered_list


# 2-6 过滤次新股
def filter_new_stock(context, stock_list):
    """
    过滤上市不足375天的次新股
    """
    if not stock_list:  # 如果列表为空直接返回
        return []

    yesterday = context.previous_date
    filtered_list = []

    for stock in stock_list:
        # 获取股票上市日期
        start_date = get_security_info(stock).start_date
        # 计算上市天数
        listing_days = (yesterday - start_date).days

        if listing_days >= 375:  # 上市超过375天
            filtered_list.append(stock)
        else:
            print(f"股票{stock}上市{listing_days}天，属于次新股，被过滤")

    return filtered_list


# 调仓函数
def monthly_adjustment(context):
    """
    根据市场风格信号调整持仓
    """
    print(f"\n=== 开始调仓 ===")
    print(f"当前市场信号: {g.singal}")

    # 获取目标持仓列表
    target_list = []
    if g.singal == 'big':
        target_list = White_Horse(context)
        print("选择大市值策略")
    elif g.singal == 'small':
        target_list = SMALL(context, [])  # 传入空列表，让SMALL函数自己获取股票池
        print("选择小市值策略")
    elif g.singal == 'etf':
        target_list = select_etf(context)
        print("选择ETF策略")
    else:
        print("未知的市场信号")
        return

    print(f"选股结果: {target_list}")

    # 过滤停牌、涨跌停股票
    target_list = filter_paused_stock(target_list)
    target_list = filter_limitup_stock(context, target_list)
    target_list = filter_limitdown_stock(context, target_list)

    print(f"过滤后的目标持仓: {target_list}")

    # 卖出不在目标列表的股票
    positions = context.portfolio.positions
    for stock in list(positions.keys()):
        if stock not in target_list and stock not in g.yesterday_HL_list:
            order_target_value(stock, 0)
            print(f"卖出股票: {stock}")

    # 计算买入金额
    position_count = len(positions)
    target_num = len(target_list)
    remaining_positions = g.buy_stock_count - position_count

    if target_num > 0 and remaining_positions > 0:
        available_cash = context.portfolio.cash
        if available_cash <= 0:
            print("没有可用资金")
            return

        # 计算每只股票的目标买入金额
        target_value = available_cash / min(remaining_positions, target_num)

        # 确保买入金额合理
        if target_value <= 0 or np.isinf(target_value):
            print(f"买入金额异常: {target_value}, 可用资金: {available_cash}, 剩余仓位: {remaining_positions}")
            return

        # 买入股票
        print(f"\n开始买入，每只股票金额: {target_value:.2f}")
        for stock in target_list:
            if stock not in positions:
                order_target_value(stock, target_value)
                print(f"买入股票: {stock}, 金额: {target_value:.2f}")
                if len(context.portfolio.positions) >= g.buy_stock_count:
                    break

    print("=== 调仓结束 ===\n")


# 2-4 过滤涨停的股票
def filter_limitup_stock(context, stock_list):
    """
    过滤涨停的股票
    """
    if not stock_list:  # 如果列表为空直接返回
        return []

    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    current_data = get_current_data()

    filtered_list = []
    for stock in stock_list:
        if stock in context.portfolio.positions.keys() or \
                last_prices[stock][-1] < current_data[stock].high_limit:
            filtered_list.append(stock)
        else:
            print(f"股票{stock}涨停，被过滤")
    return filtered_list


# 2-5 过滤跌停的股票
def filter_limitdown_stock(context, stock_list):
    """
    过滤跌停的股票
    """
    if not stock_list:  # 如果列表为空直接返回
        return []

    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    current_data = get_current_data()

    filtered_list = []
    for stock in stock_list:
        if stock in context.portfolio.positions.keys() or \
                last_prices[stock][-1] > current_data[stock].low_limit:
            filtered_list.append(stock)
        else:
            print(f"股票{stock}跌停，被过滤")
    return filtered_list


def White_Horse(context):
    """
    大市值策略选股函数
    根据市场温度选择不同的选股条件
    """
    print("\n=== 大市值选股开始 ===")
    Market_temperature(context)
    print(f"当前市场温度: {g.market_temperature}")

    # 获取沪深300成分股
    all_stocks = get_index_stocks("000300.XSHG")
    print(f"沪深300成分股数量: {len(all_stocks)}")

    # 基础过滤
    all_stocks = filter_new_stock(context, all_stocks)
    all_stocks = filter_kcbj_stock(all_stocks)
    all_stocks = filter_st_stock(all_stocks)
    all_stocks = filter_paused_stock(all_stocks)
    print(f"过滤后剩余股票数量: {len(all_stocks)}")

    # 根据市场温度设置不同的选股条件
    if g.market_temperature == "cold":
        print("\n冷市选股条件：")
        print("- PB < 1")
        print("- 经营性现金流 > 0")
        print("- 净利润 > 0")
        print("- 现金流/净利润 > 2")
        print("- ROE > 1.5%")
        print("- 净利润同比 > -15%")

        q = query(
            valuation.code
        ).filter(
            valuation.pb_ratio > 0,
            valuation.pb_ratio < 1,
            cash_flow.subtotal_operate_cash_inflow > 0,
            indicator.adjusted_profit > 0,
            cash_flow.subtotal_operate_cash_inflow / indicator.adjusted_profit > 2.0,
            indicator.inc_return > 1.5,
            indicator.inc_net_profit_year_on_year > -15,
            valuation.code.in_(all_stocks)
        ).order_by(
            (indicator.roa / valuation.pb_ratio).desc()
        ).limit(
            g.buy_stock_count + 1
        )

    elif g.market_temperature == "warm":
        print("\n温和市选股条件：")
        print("- PB < 1")
        print("- 经营性现金流 > 0")
        print("- 净利润 > 0")
        print("- 现金流/净利润 > 1")
        print("- ROE > 2%")
        print("- 净利润同比 > 0%")

        q = query(
            valuation.code
        ).filter(
            valuation.pb_ratio > 0,
            valuation.pb_ratio < 1,
            cash_flow.subtotal_operate_cash_inflow > 0,
            indicator.adjusted_profit > 0,
            cash_flow.subtotal_operate_cash_inflow / indicator.adjusted_profit > 1.0,
            indicator.inc_return > 2.0,
            indicator.inc_net_profit_year_on_year > 0,
            valuation.code.in_(all_stocks)
        ).order_by(
            (indicator.roa / valuation.pb_ratio).desc()
        ).limit(
            g.buy_stock_count + 1
        )

    else:  # hot market
        print("\n热市选股条件：")
        print("- PB > 3")
        print("- 经营性现金流 > 0")
        print("- 净利润 > 0")
        print("- 现金流/净利润 > 0.5")
        print("- ROE > 3%")
        print("- 净利润同比 > 20%")

        q = query(
            valuation.code
        ).filter(
            valuation.pb_ratio > 3,
            cash_flow.subtotal_operate_cash_inflow > 0,
            indicator.adjusted_profit > 0,
            cash_flow.subtotal_operate_cash_inflow / indicator.adjusted_profit > 0.5,
            indicator.inc_return > 3.0,
            indicator.inc_net_profit_year_on_year > 20,
            valuation.code.in_(all_stocks)
        ).order_by(
            indicator.roa.desc()
        ).limit(
            g.buy_stock_count + 1
        )

    # 获取选股结果
    df = get_fundamentals(q)
    check_out_lists = list(df.code)
    print(f"\n选出股票: {check_out_lists}")
    print("=== 大市值选股结束 ===\n")

    return check_out_lists


def Market_temperature(context):
    """
    判断市场温度
    计算沪深300指数的市场高度和波动情况
    """
    index300 = attribute_history('000300.XSHG', 220, '1d', ('close'), df=False)['close']
    market_height = (np.mean(index300[-5:]) - np.min(index300)) / (np.max(index300) - np.min(index300))

    print(f"\n=== 市场温度判断 ===")
    print(f"市场高度: {market_height:.2%}")

    if market_height < 0.20:
        g.market_temperature = "cold"
        print("判断结果：冷市")
    elif market_height > 0.90:
        g.market_temperature = "hot"
        print("判断结果：热市")
    elif np.max(index300[-60:]) / np.min(index300) > 1.20:
        g.market_temperature = "warm"
        print("判断结果：温和市")

    # 记录温度值用于可视化
    if context.run_params.type != 'sim_trade':
        temp = 200 if g.market_temperature == "cold" else \
            300 if g.market_temperature == "warm" else 400
        record(temp=temp)

    print("=== 市场温度判断结束 ===\n")
