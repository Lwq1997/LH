# 导入函数库
# -*- coding: utf-8 -*-
# 如果你的文件包含中文, 请在文件的第一行使用上面的语句指定你的文件编码

# 用到策略及数据相关API请加入下面的语句(如果要兼容研究使用可以使用 try except导入
from kuanke.user_space_api import *
from jqdata import *
from kuanke.wizard import *
import numpy as np
import pandas as pd
import datetime as dt
from jqlib.technical_analysis import *
import requests
from prettytable import PrettyTable
import inspect

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
    log.set_level('strategy', 'info')
    # 关闭未来函数
    set_option('avoid_future_data', True)

    ### 股票相关设定 ###
    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.0005, open_commission=0.0001, close_commission=0.0001, min_commission=0),
                   type='stock')

    # 为股票设定滑点为百分比滑点
    set_slippage(FixedSlippage(0.01), type='stock')

    # 临时变量

    # 持久变量
    g.strategys = {}
    # 子账户 分仓
    g.portfolio_value_proportion = [0, 0, 0, 0, 0.5, 0.5]

    # 创建策略实例
    # 初始化策略子账户 subportfolios
    set_subportfolios([
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[0], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[1], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[2], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[3], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[4], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[5], 'stock'),
    ])

    # 是否发送微信消息，回测环境不发送，模拟环境发送
    context.is_send_wx_message = 0

    params = {
        'max_hold_count': 100,  # 最大持股数
        'max_select_count': 100,  # 最大输出选股数
    }
    sbgk_strategy = SBGK_Strategy_V3(context, subportfolio_index=0, name='首板高开', params=params)
    g.strategys[sbgk_strategy.name] = sbgk_strategy

    params = {
        'max_hold_count': 100,  # 最大持股数
        'max_select_count': 100,  # 最大输出选股数
    }
    rzq_strategy = RZQ_Strategy_V3(context, subportfolio_index=1, name='弱转强', params=params)
    g.strategys[rzq_strategy.name] = rzq_strategy

    params = {
        'max_hold_count': 100,  # 最大持股数
        'max_select_count': 100,  # 最大输出选股数
    }
    sbdk_strategy = SBDK_Strategy_V3(context, subportfolio_index=2, name='首板低开', params=params)
    g.strategys[sbdk_strategy.name] = sbdk_strategy

    params = {
        'max_hold_count': 100,  # 最大持股数
        'max_select_count': 100,  # 最大输出选股数
    }
    ogt_strategy = OGT_Strategy(context, subportfolio_index=3, name='一进二', params=params)
    g.strategys[ogt_strategy.name] = ogt_strategy

    params = {
        'max_hold_count': 100,  # 最大持股数
        'max_select_count': 100,  # 最大输出选股数
    }
    total_strategy = Strategy(context, subportfolio_index=4, name='统筹交易策略', params=params)
    g.strategys[total_strategy.name] = total_strategy

    params = {
        'max_hold_count': 5,  # 最大持股数
        'max_select_count': 10,  # 最大输出选股数
        'buy_strategy_mode': 'equal'
    }
    st_strategy = ST_Strategy(context, subportfolio_index=5, name='ST策略', params=params)
    g.strategys[st_strategy.name] = st_strategy


# 模拟盘在每天的交易时间结束后会休眠，第二天开盘时会恢复，如果在恢复时发现代码已经发生了修改，则会在恢复时执行这个函数。 具体的使用场景：可以利用这个函数修改一些模拟盘的数据。
def after_code_changed(context):  # 输出运行时间
    log.info('函数运行时间(after_code_changed)：' + str(context.current_dt.time()))

    # 是否发送微信消息，回测环境不发送，模拟环境发送
    context.is_send_wx_message = 0

    unschedule_all()  # 取消所有定时运行

    run_daily(prepare_stock_list, time='09:00')

    if g.portfolio_value_proportion[4] > 0:
        # 选股
        run_daily(total_select, time='09:27')
        run_daily(total_buy, time='09:28')
        run_daily(total_sell, time='11:25')
        run_daily(total_sell, time='14:50')
        # run_daily(after_market_close, 'after_close')

    if g.portfolio_value_proportion[5] > 0:
        # 选股
        run_daily(st_select, time='09:27')
        run_daily(st_buy, time='09:28')
        # run_daily(st_sell, time='10:00')
        run_daily(st_sell, time='13:00')
        run_daily(st_sell, time='14:00')
        # run_daily(after_market_close, 'after_close')

    run_daily(after_market_close, 'after_close')


def after_market_close(context):
    # g.strategys['首板高开'].after_market_close(context)
    # g.strategys['弱转强'].after_market_close(context)
    # g.strategys['首板低开'].after_market_close(context)
    # g.strategys['一进二'].after_market_close(context)
    g.strategys['统筹交易策略'].after_market_close(context)
    g.strategys['ST策略'].after_market_close(context)


def prepare_stock_list(context):
    log.info('--prepare_stock_list选股函数--',
             str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
    utilstool = UtilsToolClass()
    utilstool.name = '总策略'
    g.fengban_rate = 0

    # 文本日期
    date = context.previous_date

    date_3, date_2, date_1, date = get_trade_days(end_date=date, count=4)

    # 初始列表
    initial_list = utilstool.stockpool(context, is_filter_highlimit=False,
                                       is_filter_lowlimit=False, is_updown_limit=False)

    # 昨日涨停
    yes_hl_list = utilstool.get_hl_stock(context, initial_list, date)
    # 前日涨停
    yes_yes_hl_list = utilstool.get_hl_stock(context, initial_list, date_1)
    # 昨日曾涨停过（包含涨停+涨停炸板）
    hl0_list = utilstool.get_ever_hl_stock(context, initial_list, date)
    # 前日曾涨停过（包含涨停+涨停炸板）
    hl1_list = utilstool.get_ever_hl_stock(context, initial_list, date_1)
    # 前前日曾涨停过（包含涨停+涨停炸板）
    hl2_list = utilstool.get_ever_hl_stock(context, initial_list, date_2)
    # 合并 hl1_list 和 hl2_list 为一个集合，用于快速查找需要剔除的元素
    elements_to_remove = set(hl1_list + hl2_list)
    # log.info('initial_list:', '001287.XSHE' in initial_list)
    # log.info('yes_hl_list:', '001287.XSHE' in yes_hl_list)
    # log.info('hl1_list:', '001287.XSHE' in hl1_list)
    # log.info('hl2_list:', '001287.XSHE' in hl2_list)
    # 昨日涨停，但是前2天都没有涨停过，真昨日首板
    context.yes_first_hl_list = [stock for stock in yes_hl_list if stock not in elements_to_remove]
    # 昨日涨停，但是前1天都没有涨停过
    context.yes_no_first_hl_list = [stock for stock in yes_hl_list if stock not in hl1_list]

    # 昨日曾涨停炸板
    h0_list = utilstool.get_ever_hl_stock2(context, initial_list, date)
    h1_list = utilstool.get_ever_hl_stock2(context, initial_list, date_1)
    h2_list = utilstool.get_ever_hl_stock2(context, initial_list, date_2)
    h3_list = utilstool.get_ever_hl_stock2(context, initial_list, date_3)
    # 上上个交易日涨停
    elements_to_remove2 = utilstool.get_hl_stock(context, initial_list, date_1)

    # 过滤上上个交易日涨停、曾涨停
    context.yes_first_no_hl_list = [stock for stock in h0_list if stock not in elements_to_remove2]

    two_hl_list = list(set(yes_hl_list) & set(yes_yes_hl_list))  # 取交集，确保两个交易日均涨停
    early_two_remove = list(set(h2_list) | set(h3_list))  # 取并集，确保前2个交易日没有涨停过
    early_three_remove = list(set(h1_list) | set(h2_list))  # 取并集，确保前2个交易日没有涨停过

    context.two_hl_list = [stock for stock in two_hl_list if stock not in early_two_remove]
    context.three_hl_list = [stock for stock in yes_hl_list if stock not in early_three_remove]

    if len(yes_hl_list) > 0 and len(hl0_list) > 0:
        log.info(
            f'{date}的涨停家数{len(yes_hl_list)},涨停过的股票数{len(hl0_list)},封板率是{len(yes_hl_list) / len(hl0_list)}'
        )
        g.fengban_rate = len(yes_hl_list) / len(hl0_list)


def total_select(context):
    g.strategys['弱转强'].select(context)
    g.strategys['首板高开'].select(context)
    g.strategys['首板低开'].select(context)
    g.strategys['一进二'].select(context)
    total_stocks = set(
        g.strategys['弱转强'].select_list
        + g.strategys['首板高开'].select_list
        + g.strategys['首板低开'].select_list
        + g.strategys['一进二'].select_list
    )
    g.strategys['统筹交易策略'].special_select_list = {}
    if total_stocks:
        g.strategys['统筹交易策略'].special_select_list = firter_industry(context, total_stocks)


def firter_industry(context, total_stocks):
    log.info('--firter_industry函数--',
             str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
    if g.fengban_rate < 0.5:
        return {}  # 返回空列表或其他默认值
    first_filter_stocks = []
    concept_final_stocks = []
    industry_final_stocks = []
    industry_qualified_stocks = []
    concept_qualified_stocks = []
    # 前置过滤
    for s in total_stocks:
        # history_data = attribute_history(s, 20, '1d', ['close', 'volume'], skip_paused=True)
        # # 条件1：股价在20日均线上方，且短期均线多头排列
        # ma5 = history_data['close'][-5:].mean()
        # ma10 = history_data['close'][-10:].mean()
        # ma20 = history_data['close'].mean()
        # if not (ma5 > ma10 > ma20 and history_data['close'][-1] > ma20):
        #     continue
        first_filter_stocks.append(s)

    if first_filter_stocks:
        # 获取最近 5 天的行业热度
        start_date = (context.current_dt.date() - dt.timedelta(days=6)).strftime('%Y-%m-%d')
        end_date = context.previous_date.strftime('%Y-%m-%d')
        industry_heat_df, concept_heat_df = get_industry_concpet_heat(context, start_date, end_date)
        # 获取行业热度排名前5的行业
        top_industries = industry_heat_df.head(5)["行业"].tolist()
        log.info("行业热度排名前五的行业：", top_industries)

        for s in total_stocks:
            # 获取股票所属行业
            stock_industry = get_industry(s, date=context.current_dt.date())
            industry_name = stock_industry[s]["sw_l1"]["industry_name"]
            log.info(f'当前股票{s}所属行业{industry_name}')
            # 如果股票所属行业在热度排名前三的行业中，则加入选股列表
            if industry_name in top_industries:
                industry_qualified_stocks.append(s)

        # 获取行业热度排名前10的概念
        top_concepts = concept_heat_df.head(5)["概念"].tolist()
        log.info("概念热度排名前5的概念：", top_concepts)

        for s in total_stocks:
            # 获取股票所属概念
            stock_concept = get_concept(s, date=context.current_dt.date())
            concept_names = stock_concept[s]["jq_concept"]
            log.info(f'当前股票{s}所属概念{concept_names}')
            for concept in concept_names:
                # 如果股票所属行业在热度排名前三的行业中，则加入选股列表
                if concept['concept_name'] in top_concepts:
                    concept_qualified_stocks.append(s)
        concept_final_stocks = set(concept_qualified_stocks)
        industry_final_stocks = set(industry_qualified_stocks)

    special_select_list = {
        '行业': industry_final_stocks,
        '概念': concept_final_stocks
    }
    print("今日最终选股: " + str(special_select_list))
    # 将选股结果存储到全局变量
    return special_select_list


# 获取指定日期范围内的行业热度
def get_industry_concpet_heat(context, start_date, end_date):
    industry_heat_dict = {}  # 存储行业热度的字典
    concept_heat_dict = {}  # 存储概念热度的字典
    date_range = pd.date_range(start=start_date, end=end_date)

    for search_date in date_range:
        search_date = search_date.strftime('%Y-%m-%d')
        # 获取当天涨停的股票
        today_limit_stocks = get_today_limit_stocks(context, search_date)
        # 获取股票所属行业
        stock_industry_df = get_stock_industry_df(context, search_date, today_limit_stocks)
        # 获取股票所属概念
        stock_concept_df = get_stock_concept_df(context, search_date, today_limit_stocks)
        # 统计每个行业的涨停股数量
        stock_industry_df["涨停数量"] = 1
        industry_count_df = stock_industry_df.groupby(["sw_L1"]).count()
        industry_count_df = industry_count_df.drop(["code", "sw_L2", "sw_L3"], axis=1)

        # 累加行业热度
        for industry, count in industry_count_df.iterrows():
            if industry in industry_heat_dict:
                industry_heat_dict[industry] += count["涨停数量"]
            else:
                industry_heat_dict[industry] = count["涨停数量"]

        # 统计每个概念的涨停股数量
        stock_concept_df["涨停数量"] = 1
        concept_count_df = stock_concept_df.groupby(["concept"]).count()

        # 累加行业热度
        for concept, count in concept_count_df.iterrows():
            if concept in concept_heat_dict:
                concept_heat_dict[concept] += count["涨停数量"]
            else:
                concept_heat_dict[concept] = count["涨停数量"]
    # 将行业热度字典转换为 DataFrame
    industry_heat_df = pd.DataFrame(list(industry_heat_dict.items()), columns=["行业", "涨停总数"])
    industry_heat_df = industry_heat_df.sort_values(by="涨停总数", ascending=False)

    # 将概念热度字典转换为 DataFrame
    concept_heat_df = pd.DataFrame(list(concept_heat_dict.items()), columns=["概念", "涨停总数"])
    concept_heat_df = concept_heat_df.sort_values(by="涨停总数", ascending=False)

    print('行业热度:', industry_heat_df)
    print('概念热度:', concept_heat_df)

    return industry_heat_df, concept_heat_df


# 获取当天涨停的股票，过滤掉上市不足半年的票
def get_today_limit_stocks(context, search_date):
    # 获取所有股票，排除上市不足半年
    all_stocks = get_all_securities(types=['stock'], date=search_date)
    # 半年前的日期
    pre_half_year_date = dt.datetime.strptime(search_date, "%Y-%m-%d") - dt.timedelta(days=180)
    pre_half_year_date = pre_half_year_date.date()
    # 过滤上市不足半年的股票
    all_stocks = all_stocks[all_stocks['start_date'] < pre_half_year_date]

    # 获取当天的收盘价和涨停价
    today_df = get_price(list(all_stocks.index), end_date=search_date, count=1, frequency='1d',
                         fields=['close', 'high_limit'], panel=False, fill_paused=False)
    # 筛选出当天涨停的股票
    today_limit_df = today_df[today_df['close'] == today_df['high_limit']]

    return list(today_limit_df.code)


# 获取股票所属行业
def get_stock_industry_df(context, search_date, stocks):
    stock_industry_dict = get_industry(stocks, date=search_date)
    # 申万一、二、三级行业
    sw_L1 = []
    sw_L2 = []
    sw_L3 = []

    for stock in stocks:
        industry_dict = stock_industry_dict[stock]
        sw_L1.append(industry_dict["sw_l1"]["industry_name"])
        sw_L2.append(industry_dict["sw_l2"]["industry_name"])
        sw_L3.append(industry_dict["sw_l3"]["industry_name"])

    stock_industry_df = pd.DataFrame(columns=['code', 'sw_L1', 'sw_L2', 'sw_L3'])
    stock_industry_df["code"] = stocks
    stock_industry_df["sw_L1"] = sw_L1
    stock_industry_df["sw_L2"] = sw_L2
    stock_industry_df["sw_L3"] = sw_L3

    return stock_industry_df


# 获取股票所属概念
def get_stock_concept_df(context, search_date, stocks):
    stock_concpet_dict = get_concept(stocks, date=search_date)
    # 聚宽概念
    data = []
    black_concept_name = [
        '转融券标的',
        '融资融券',
        '国企改革',
        '深股通',
        '沪股通'
    ]

    for stock in stocks:
        jq_concept_dict = stock_concpet_dict[stock]['jq_concept']
        for concept in jq_concept_dict:
            if concept['concept_name'] not in black_concept_name:
                data.append({'code': stock, 'concept': concept['concept_name']})
    # 创建多行DataFrame
    stock_concept_df = pd.DataFrame(data)
    # log.info(search_date, '日，概念涨停明细:', stock_concept_df)

    return stock_concept_df


def total_buy(context):
    g.strategys['统筹交易策略'].specialBuy(context, split=999)


def total_sell(context):
    g.strategys['统筹交易策略'].specialSell(context)


def st_select(context):
    g.strategys['ST策略'].select(context)


def st_buy(context):
    g.strategys['ST策略'].specialBuy(context, split=3)


def st_sell(context):
    g.strategys['ST策略'].specialSell(context, is_st_sell=True)

class UtilsToolClass:
    def __init__(self):
        self.name = None
        self.subportfolio_index = None

    def set_params(self, name, subportfolio_index):
        self.name = name
        self.subportfolio_index = subportfolio_index

    # 计算股票处于一段时间内相对位置
    def get_relative_position_df(self, context, stock_list, date, watch_days):
        if len(stock_list) != 0:
            df = get_price(stock_list, end_date=date, fields=['high', 'low', 'close'], count=watch_days,
                           fill_paused=False,
                           skip_paused=False, panel=False).dropna()
            close = df.groupby('code').apply(lambda df: df.iloc[-1, -1])
            high = df.groupby('code').apply(lambda df: df['high'].max())
            low = df.groupby('code').apply(lambda df: df['low'].min())
            result = pd.DataFrame()
            result['rp'] = (close - low) / (high - low)
            return result
        else:
            return pd.DataFrame(columns=['rp'])

    def rise_low_volume(self, context, stock):  # 上涨时，未放量 rising on low volume
        hist = attribute_history(stock, 106, '1d', fields=['high', 'volume'], skip_paused=True, df=False)
        high_prices = hist['high'][:102]
        prev_high = high_prices[-1]
        zyts_0 = next((i - 1 for i, high in enumerate(high_prices[-3::-1], 2) if high >= prev_high), 100)
        zyts = zyts_0 + 5
        if hist['volume'][-1] <= max(hist['volume'][-zyts:-1]) * 0.9:
            return True
        return False

    # 计算左压天数
    def calculate_zyts(self, context, stock):
        high_prices = attribute_history(stock, 101, '1d', fields=['high'], skip_paused=True)['high']
        prev_high = high_prices.iloc[-1]
        zyts_0 = next((i - 1 for i, high in enumerate(high_prices[-3::-1], 2) if high >= prev_high), 100)
        zyts = zyts_0 + 5
        return zyts

    def transform_date(self, context, date, date_type):
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

    def get_shifted_date(self, context, date, days, days_type='T'):
        # 获取上一个自然日
        d_date = self.transform_date(context, date, 'd')
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

    def stockpool(self, context, pool_id=1, index=None, is_filter_kcbj=True, is_filter_st=True, is_filter_paused=True,
                  is_filter_highlimit=True,
                  is_filter_lowlimit=True, is_filter_new=True, is_updown_limit=True):
        log.info(self.name, '--stockpool函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        if index is None:
            lists = list(get_all_securities(types=['stock'], date=context.previous_date).index)
        else:
            lists = list(get_index_stocks(index))

        if pool_id == 0:
            pass
        elif pool_id == 1:
            if is_filter_kcbj:
                lists = self.filter_kcbj_stock(context, lists)
            if is_filter_st:
                lists = self.filter_st_stock(context, lists, is_updown_limit=is_updown_limit)
            if is_filter_paused:
                lists = self.filter_paused_stock(context, lists)
            if is_filter_highlimit:
                lists = self.filter_highlimit_stock(context, lists)
            if is_filter_lowlimit:
                lists = self.filter_lowlimit_stock(context, lists)
            if is_filter_new:
                lists = self.filter_new_stock(context, lists, days=375)

        return lists

    ##################################  交易函数群 ##################################

    # 开仓单只
    def open_position(self, context, security, value, target=True):
        now = str(context.current_dt.date()) + ' ' + str(context.current_dt.time())
        now_time = context.current_dt.time()
        current_data = get_current_data()
        before_buy = dt.time(9, 30) >= now_time
        # log.info('before_buy:',before_buy)
        style_arg = MarketOrderStyle(current_data[security].day_open) if before_buy else None
        if target:
            order_info = order_target_value(security, value, style=style_arg, pindex=self.subportfolio_index)
        else:
            # log.info('S:', security, "--value:", value)
            order_info = order_value(security, value, style=style_arg, pindex=self.subportfolio_index)

        method_name = inspect.getframeinfo(inspect.currentframe()).function
        item = f"分仓策略:{self.name}<br>-函数名称:{method_name}<br>-时间:{now}"
        if order_info != None and (before_buy or order_info.filled > 0):
            content = (f"策略: {self.name} "
                       f"--操作时间: {now} "
                       f"--买入股票: {security} "
                       f"--计划买入金额: {value} "
                       f"--买入数量: {order_info.amount} "
                       f"--成交数量: {order_info.filled} "
                       f"--买入均价: {order_info.price} "
                       f"--实际买入金额: {order_info.price * order_info.filled} "
                       f"--交易佣金: {order_info.commission:.2f}\n<br>")
            log.info(content)
            send_message(content)
            self.send_wx_message(context, item, content)
            return True
        content = (f"策略: {self.name} "
                   f"--操作时间: {now} "
                   f"--买入股票，交易失败！！股票: {security} "
                   f"--失败原因: {order_info} "
                   f"--计划买入金额: {value}\n<br>")
        log.error(content)
        send_message(content)
        self.send_wx_message(context, item, content)
        return False

    # 清仓单只
    def close_position(self, context, security, value, target=True):
        now = str(context.current_dt.date()) + ' ' + str(context.current_dt.time())
        if target:
            order_info = order_target_value(security, value, pindex=self.subportfolio_index)
        else:
            order_info = order_value(security, value, pindex=self.subportfolio_index)
        method_name = inspect.getframeinfo(inspect.currentframe()).function
        item = f"分仓策略:{self.name}<br>-函数名称:{method_name}<br>-时间:{now}"

        if order_info != None and order_info.status == OrderStatus.held and order_info.filled == order_info.amount:
            # 计算收益率:（当前价格/持仓价格）- 1
            ret = 100 * (order_info.price / order_info.avg_cost - 1)
            # 计算收益金额: 可卖仓位 *（当前价格/持仓价格)
            ret_money = order_info.amount * (order_info.price - order_info.avg_cost)
            content = (f"策略: {self.name} "
                       f"--操作时间: {now} "
                       f"--卖出股票: {security} "
                       f"--卖出数量: {order_info.amount} "
                       f"--成交数量: {order_info.filled} "
                       f"--持仓均价: {order_info.avg_cost} "
                       f"--卖出均价: {order_info.price} "
                       f"--实际卖出金额: {order_info.price * order_info.filled} "
                       f"--交易佣金: {order_info.commission:.2f} 收益率: {ret:.2f}% 收益金额: {ret_money:.2f} \n<br>")
            log.info(content)
            send_message(content)
            self.send_wx_message(context, item, content)
            return True
        content = (f"策略: {self.name} "
                   f"--操作时间: {now} "
                   f"--失败原因: {order_info} "
                   f"--卖出股票，交易失败！！！股票: {security} \n<br>")
        log.error(content)
        send_message(content)
        self.send_wx_message(context, item, content)
        return False

    ##################################  选股函数群 ##################################

    # 获取股票股票池（暂无使用）
    def get_security_universe(self, context, security_universe_index, security_universe_user_securities):
        log.info(self.name, '--get_security_universe函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        temp_index = []
        for s in security_universe_index:
            if s == 'all_a_securities':
                temp_index += list(get_all_securities(['stock'], context.current_dt.date()).index)
            else:
                temp_index += get_index_stocks(s)
        for x in security_universe_user_securities:
            temp_index += x
        return sorted(list(set(temp_index)))

    # 过滤科创北交
    def filter_basic_stock(self, context, stock_list):
        log.info(self.name, '--filter_basic_stock过滤股票函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        current_data = get_current_data()
        return [
            stock
            for stock in stock_list
            if not current_data[stock].paused
               and not current_data[stock].is_st
               and "ST" not in current_data[stock].name
               and "*" not in current_data[stock].name
               and "退" not in current_data[stock].name
               and not (
                    stock.startswith('4') or
                    stock.startswith('8') or
                    stock.startswith('68') or
                    stock.startswith('30')
            )
               and not context.previous_date - get_security_info(stock).start_date < dt.timedelta(days=375)

        ]
        return stock_list

    # 过滤科创北交
    def filter_kcbj_stock(self, context, stock_list):
        log.info(self.name, '--filter_kcbj_stock过滤科创北交函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 使用列表推导式过滤股票
        filtered_stock_list = [stock for stock in stock_list if not (stock.startswith('4') or
                                                                     stock.startswith('8') or
                                                                     stock.startswith('68') or
                                                                     stock.startswith('30'))]

        return filtered_stock_list

    # 过滤N天卖出的涨停股
    def filter_recently_sold(self, context, stocks, diff_day):
        log.info(self.name, '--filter_recently_sold过滤最近卖出股票--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()),
                 '--历史所有卖出股票如下--', g.global_sold_stock_record)
        current_date = context.current_dt.date()
        global_sold_stock_record = g.global_sold_stock_record
        return [stock for stock in stocks if
                stock not in global_sold_stock_record or (
                        current_date - global_sold_stock_record[stock]).days >= diff_day]

    # 过滤停牌股票
    def filter_paused_stock(self, context, stock_list):
        log.info(self.name, '--filter_paused_stock过滤停牌股票函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        current_data = get_current_data()
        return [stock for stock in stock_list if not current_data[stock].paused]

    # 过滤ST及其他具有退市标签的股票
    def filter_st_stock(self, context, stock_list, is_updown_limit=True):
        log.info(self.name, '--filter_st_stock过滤ST及其他具有退市标签的股票函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        current_data = get_current_data()
        result_list = [stock for stock in stock_list
                       if not current_data[stock].is_st
                       and 'ST' not in current_data[stock].name
                       and '*' not in current_data[stock].name
                       and '退' not in current_data[stock].name]
        if is_updown_limit:
            result_list = [stock for stock in result_list
                           if not current_data[stock].last_price >= current_data[stock].high_limit * 0.97
                           and not current_data[stock].last_price <= current_data[stock].low_limit * 1.04
                           ]
        return result_list

    # 过滤涨停的股票
    def filter_highlimit_stock(self, context, stock_list):
        log.info(self.name, '--filter_highlimit_stock过滤涨停的股票函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        subportfolio = context.subportfolios[self.subportfolio_index]
        last_prices = history(1, unit='1m', field='close', security_list=stock_list)
        current_data = get_current_data()

        return [stock for stock in stock_list if stock in subportfolio.long_positions
                or current_data[stock].last_price < current_data[stock].high_limit]

    # 过滤跌停的股票
    def filter_lowlimit_stock(self, context, stock_list):
        log.info(self.name, '--filter_lowlimit_stock过滤跌停的股票函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        subportfolio = context.subportfolios[self.subportfolio_index]
        last_prices = history(1, unit='1m', field='close', security_list=stock_list)
        current_data = get_current_data()

        for stock in stock_list:
            log.debug(
                f'股票{stock},当前最新价格{last_prices[stock][-1]},当前跌停价{current_data[stock].low_limit},,当前涨停价{current_data[stock].high_limit}')
        return [stock for stock in stock_list if stock in subportfolio.long_positions
                or current_data[stock].last_price > current_data[stock].low_limit]

    # 过滤次新股（小市值专用）
    def filter_new_stock(self, context, stock_list, days):
        log.info(self.name, '--filter_new_stock过滤次新股函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        return [stock for stock in stock_list if
                not context.previous_date - get_security_info(stock).start_date < dt.timedelta(days=days)]

    # 行业股票数量限制
    def filter_stocks_by_industry(self, context, stocks, max_industry_stocks):
        log.info(self.name, '--filter_stocks_by_industry函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        industry_info = self.getStockIndustry(stocks)
        log.info('本次选股的行业信息是:', industry_info)
        selected_stocks = []
        industry_count = {}
        for stock in stocks:
            industry = industry_info[stock]
            if industry not in industry_count:
                industry_count[industry] = 0
            if industry_count[industry] < max_industry_stocks:
                selected_stocks.append(stock)
                industry_count[industry] += 1
        return selected_stocks

    # 过滤大幅解禁（小市值专用）
    def filter_locked_shares(self, context, stock_list, days):
        log.info(self.name, '--filter_locked_shares过滤解禁股函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 获取指定日期区间内的限售解禁数据
        df = get_locked_shares(stock_list=stock_list, start_date=context.previous_date.strftime('%Y-%m-%d'),
                               forward_count=days)
        # 过滤出解禁数量占总股本的百分比超过 20% 的股票
        df = df[df['rate1'] > 0.2]
        filterlist = list(df['code'])
        # 从股票池中排除这些股票
        return [stock for stock in stock_list if stock not in filterlist]

    ###################################  公用函数群 ##################################
    # 获取个股行业,暂无使用
    def get_industry_name(self, i_Constituent_Stocks, value):
        return [k for k, v in i_Constituent_Stocks.items() if value in v]

    # 把prettytable对象转换成键值对字符串
    def pretty_table_to_kv_string(self, table):
        headers = table.field_names
        result = ""
        data_rows = table._rows  # 直接获取表格内部存储的数据行列表，避免格式干扰
        for row in data_rows:
            for header, cell in zip(headers, row):
                result += f"{header}: {cell}\n<br>"
            result += "\n<br>"
        return result.rstrip()

    # 发送微信消息
    def send_wx_message(self, context, item, message):
        if context.is_send_wx_message != 1:
            return
        url = "https://wxpusher.zjiecode.com/api/send/message"

        data = {
            "appToken": "AT_B7CVGazuAWXoqBoIlGAzlIwkunQuXIQM",
            "content": f"<h1>{item}</h1><br/><p style=\"color:red;\">{message}</p>",
            "summary": item,
            "contentType": 2,
            "topicIds": [
                36105
            ],
            "url": "https://wxpusher.zjiecode.com",
            "verifyPay": False,
            "verifyPayType": 0
        }
        response = requests.post(url, json=data)
        # 可以根据需要查看响应的状态码、内容等信息
        # print(response.status_code)
        # print(response.text)

    # 计算市场宽度
    def get_market_breadth(self, context, max_industry_cnt):
        log.info(self.name, '--get_market_breadth--计算市场宽度，选择偏离程度最高的行业--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        """
        计算市场宽度，选择偏离程度最高的行业
        """
        # 指定日期以防止未来数据
        yesterday = context.previous_date

        # 获取上证中小盘指数的成分股
        stocks = get_index_stocks("000985.XSHG")

        # 获取历史收盘价数据，包括20天移动平均所需的数据
        count = 1
        historical_prices = get_price(
            stocks,
            end_date=yesterday,
            frequency="1d",
            fields=["close"],
            count=count + 20,
            panel=False,
        )

        # 将时间字段转换为日期
        historical_prices["date"] = pd.DatetimeIndex(historical_prices['time']).date

        # 将数据重塑为股票代码为索引，日期为列
        close_prices = historical_prices.pivot(index="code", columns="date", values="close")
        close_prices = close_prices.dropna(axis=0)

        # 计算20日移动平均
        ma20 = close_prices.rolling(window=20, axis=1).mean().iloc[:, -count:]

        # 获取最新一天的收盘价
        last_close_prices = close_prices.iloc[:, -count:]

        # 计算偏离程度（当前收盘价是否大于20日均线）
        bias = last_close_prices > ma20

        # 获取股票所属行业
        industries = self.getStockIndustry(stocks)
        bias["industry_name"] = industries

        # 按行业统计偏离股票的比例
        industry_bias_sum = bias.groupby("industry_name").sum()
        industry_bias_count = bias.groupby("industry_name").count()
        df_ratio = (industry_bias_sum * 100.0 / industry_bias_count).round()

        # 获取偏离比例最高的行业
        top_values = df_ratio.loc[:, yesterday].nlargest(max_industry_cnt)
        top_industries = top_values.index.tolist()

        # 计算全市场宽度的平均偏离比例
        market_width = df_ratio.sum(axis=0).mean()

        log.info(
            [name for name in top_industries],
            "  全市场宽度：",
            market_width
        )

        return top_industries

    def getStockIndustry(self, stocks):
        industry = get_industry(stocks)
        dict = {
            stock: info["sw_l1"]["industry_name"]
            for stock, info in industry.items()
            if "sw_l1" in info
        }
        return pd.Series(dict)

    # 计算市场温度
    def Market_temperature(self, context, market_temperature='warm'):
        log.info(self.name, '--Market_temperature函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        # 获取数据：使用attribute_history函数获取沪深300指数过去220天的收盘价数据。
        index300 = attribute_history('000300.XSHG', 220, '1d', ('close'), df=False)['close']

        # 计算市场高度：通过计算最近5天收盘价的平均值与过去220天收盘价的最小值之差，再除以过去220天收盘价的最大值与最小值之差，得到市场高度（market_height)。
        market_height = (mean(index300[-5:]) - min(index300)) / (max(index300) - min(index300))

        # 判断市场温度：根据市场高度的值，将市场温度分为三种状态：
        # ·如果市场高度小于0.20, 则市场温度为"cold"。
        # ·如果市场高度大于0.90, 则市场温度为"hot"。
        # ·如果过去60天内的最高收盘价与最低收盘价之比大于1.20, 则市场温度为"warm"
        if market_height < 0.20:
            market_temperature = "cold"
        elif market_height > 0.90:
            market_temperature = "hot"
        elif max(index300[-60:]) / min(index300) > 1.20:
            market_temperature = "warm"

        return market_temperature

    # 4-1 打印每日持仓信息,暂无使用
    def print_position_info(self, context):
        log.info(self.name, '--print_position_info函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 打印当天成交记录
        trades = get_trades()
        for _trade in trades.values():
            log.info('成交记录：' + str(_trade))
        # 打印账户信息
        for position in list(context.portfolio.positions.values()):
            securities = position.security
            cost = position.avg_cost
            price = position.price
            ret = 100 * (price / cost - 1)
            value = position.value
            amount = position.total_amount
            log.info('代码:{}'.format(securities))
            log.info('成本价:{}'.format(format(cost, '.2f')))
            log.info('现价:{}'.format(price))
            log.info('收益率:{}%'.format(format(ret, '.2f')))
            log.info('持仓(股):{}'.format(amount))
            log.info('市值:{}'.format(format(value, '.2f')))
            log.info('———————————————————————————————————')
        log.info('———————————————————————————————————————分割线————————————————————————————————————————')

    # 筛选出某一日涨停的股票
    def get_hl_stock(self, context, stock_list, end_date):
        if not stock_list: return []
        h_s = get_price(stock_list, end_date=end_date, frequency='daily', fields=['close', 'high_limit', 'paused'],
                        count=1, panel=False, fill_paused=False, skip_paused=False
                        ).query('close==high_limit and paused==0').groupby('code').size()
        return h_s.index.tolist()

    # 筛选出某一日曾经涨停的股票，含炸板的
    def get_ever_hl_stock(self, context, stock_list, end_date):
        if not stock_list: return []
        h_s = get_price(stock_list, end_date=end_date, frequency='daily', fields=['high', 'high_limit', 'paused'],
                        count=1, panel=False, fill_paused=False, skip_paused=False
                        ).query('high==high_limit and paused==0').groupby('code').size()
        return h_s.index.tolist()

    # 筛选出某一日曾经涨停但未封板的股票
    def get_ever_hl_stock2(self, context, stock_list, end_date):
        if not stock_list: return []
        h_s = get_price(stock_list, end_date=end_date, frequency='daily',
                        fields=['close', 'high', 'high_limit', 'paused'],
                        count=1, panel=False, fill_paused=False, skip_paused=False
                        ).query('close!=high_limit and high==high_limit and paused==0').groupby('code').size()
        return h_s.index.tolist()

    # 筛选出某一日未涨停的股票
    def get_ever_hl_stock3(self, context, stock_list, end_date):
        if not stock_list: return []
        h_s = get_price(stock_list, end_date=end_date, frequency='daily',
                        fields=['close', 'high', 'high_limit', 'paused'],
                        count=1, panel=False, fill_paused=False, skip_paused=False
                        ).query('close!=high_limit').groupby('code').size()
        return h_s.index.tolist()

    def balance_subportfolios(self, context):
        log.info(self.name, '--balance_subportfolios平衡账户资金函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        length = len(context.portfolio_value_proportion)
        # 计算平衡前仓位比例
        log.info(
            "仓位计划调整的比例："
            + str(
                g.portfolio_value_proportion
            )
            +
            "仓位调整前的比例："
            + str(
                [
                    context.subportfolios[i].total_value / context.portfolio.total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整前的总金额："
            + str(
                [
                    context.subportfolios[i].total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整前的可用金额："
            + str(
                [
                    context.subportfolios[i].available_cash
                    for i in range(length)
                ]
            )
        )
        # 先把所有可用资金打入一号资金仓位
        for i in range(1, length):
            target = context.portfolio_value_proportion[i] * context.portfolio.total_value
            value = context.subportfolios[i].total_value
            if context.subportfolios[i].available_cash > 0 and target < value:
                transfer_cash(
                    from_pindex=i,
                    to_pindex=0,
                    cash=min(value - target, context.subportfolios[i].available_cash),
                )
        # 如果子账户仓位过低，从一号仓位往其中打入资金
        for i in range(1, length):
            target = context.portfolio_value_proportion[i] * context.portfolio.total_value
            value = context.subportfolios[i].total_value
            if target > value and context.subportfolios[0].available_cash > 0:
                transfer_cash(
                    from_pindex=0,
                    to_pindex=i,
                    cash=min(target - value, context.subportfolios[0].available_cash),
                )
        # 计算平衡后仓位比例
        log.info(
            "仓位调整后的比例："
            + str(
                [
                    context.subportfolios[i].total_value / context.portfolio.total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整后的金额："
            + str(
                [
                    context.subportfolios[i].total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整后的可用金额："
            + str(
                [
                    context.subportfolios[i].available_cash
                    for i in range(length)
                ]
            )
        )

    def balance_subportfolios_by_small_to_other(self, context):
        current_month = context.current_dt.month
        if current_month not in (1, 4):
            return
        log.info(self.name, '--balance_subportfolios_by_small_to_other 择时资金转出--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        length = len(context.portfolio_value_proportion)
        # 计算平衡前仓位比例
        log.info(
            "仓位调整前的比例："
            + str(
                [
                    context.subportfolios[i].total_value / context.portfolio.total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整前的总金额："
            + str(
                [
                    context.subportfolios[i].total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整前的可用金额："
            + str(
                [
                    context.subportfolios[i].available_cash
                    for i in range(length)
                ]
            )
        )
        if current_month in (1, 4):
            # 把小市值仓位的资金均分到大市值和ETF
            transfer_dict = {}
            available_cash = context.subportfolios[length - 1].available_cash
            for i in range(0, length - 1):
                log.info('第', length - 1, '个仓位当前可用金额:', available_cash,
                         '，按【', context.portfolio_value_proportion[i], '】比例转到仓位', i)
                value = available_cash * (context.portfolio_value_proportion[i] / 0.5)
                transfer_dict[i] = value
                if value > 0:
                    transfer_cash(
                        from_pindex=length - 1,  ## 小市值
                        to_pindex=i,  ## 大市值 && ETF
                        cash=value
                    )
                    log.info('第', length - 1, '个仓位给第', i, '个仓位转账:', value)
            context.balance_value[current_month] = transfer_dict.copy()
        # 计算平衡后仓位比例
        log.info(
            "仓位调整后的比例："
            + str(
                [
                    context.subportfolios[i].total_value / context.portfolio.total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整后的金额："
            + str(
                [
                    context.subportfolios[i].total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整后的可用金额："
            + str(
                [
                    context.subportfolios[i].available_cash
                    for i in range(length)
                ]
            )
        )

    def balance_subportfolios_by_other_to_small(self, context):
        current_month = context.current_dt.month
        if current_month not in (2, 5):
            return
        log.info(self.name, '--balance_subportfolios_by_other_to_small 择时资金转入--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        length = len(context.portfolio_value_proportion)
        # 计算平衡前仓位比例
        log.info(
            "仓位调整前的比例："
            + str(
                [
                    context.subportfolios[i].total_value / context.portfolio.total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整前的金额："
            + str(
                [
                    context.subportfolios[i].total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整前的可用金额："
            + str(
                [
                    context.subportfolios[i].available_cash
                    for i in range(length)
                ]
            )
        )
        if current_month in (2, 5):
            # 获取上个月份
            if current_month == 2:
                last_month = 1
            else:
                last_month = 4
            # 检查是否有上个月的转账记录
            if last_month in context.balance_value:
                transfer_dict = context.balance_value[last_month]
                # 把小市值仓位的资金归还回来
                for i in transfer_dict:
                    value = transfer_dict[i]
                    if value > 0:
                        transfer_cash(
                            from_pindex=i,
                            to_pindex=length - 1,  ## 小市值
                            cash=value
                        )
                        log.info('第', i, '个仓位给第', length - 1, '个仓位转账:', value)
                # 删除上个月的转账记录
                del context.balance_value[last_month]
            else:
                log.info('没有上个月的转账记录，无需归还')
        # 计算平衡后仓位比例
        log.info(
            "仓位调整后的比例："
            + str(
                [
                    context.subportfolios[i].total_value / context.portfolio.total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整后的金额："
            + str(
                [
                    context.subportfolios[i].total_value
                    for i in range(length)
                ]
            )
            +
            "仓位调整后的可用金额："
            + str(
                [
                    context.subportfolios[i].available_cash
                    for i in range(length)
                ]
            )
        )

    ##获取所有ST股##
    def get_st(self, context, is_basic = True):
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

        return filtered_df if is_basic else df

# 策略基类
class Strategy:
    def __init__(self, context, subportfolio_index, name, params):
        self.subportfolio_index = subportfolio_index
        self.name = name
        self.params = params

        self.trade_num = 0
        self.win_num = 0
        self.win_lose_rate = 0
        self.sharp = 0
        self.portfolio_value = pd.DataFrame(columns=['date', 'total_value'])
        self.strategyID = self.params['strategyID'] if 'strategyID' in self.params else ''
        self.inout_cash = 0

        self.fill_stock = self.params[
            'fill_stock'] if 'fill_stock' in self.params else '511880.XSHG'
        self.sold_diff_day = self.params[
            'sold_diff_day'] if 'sold_diff_day' in self.params else 0  # 是否过滤N天内涨停并卖出股票
        self.max_industry_cnt = self.params[
            'max_industry_cnt'] if 'max_industry_cnt' in self.params else 0  # 最大行业数
        self.buy_strategy_mode = self.params[
            'buy_strategy_mode'] if 'buy_strategy_mode' in self.params else 'equal'  # 最大持股数
        self.max_hold_count = self.params['max_hold_count'] if 'max_hold_count' in self.params else 1  # 最大持股数
        self.max_select_count = self.params['max_select_count'] if 'max_select_count' in self.params else 5  # 最大输出选股数
        self.hold_limit_days = self.params['hold_limit_days'] if 'hold_limit_days' in self.params else 20  # 计算最近持有列表的天数
        self.use_empty_month = self.params['use_empty_month'] if 'use_empty_month' in self.params else False  # 是否有空仓期
        self.empty_month = self.params['empty_month'] if 'empty_month' in self.params else []  # 空仓月份
        self.use_stoplost = self.params['use_stoplost'] if 'use_stoplost' in self.params else False  # 是否使用止损
        self.empty_month_last_day = self.params[
            'empty_month_last_day'] if 'empty_month_last_day' in self.params else []  # 需要月末清仓的月份
        self.use_empty_month_last_day = self.params[
            'use_empty_month_last_day'] if 'use_empty_month_last_day' in self.params else False  # 是否月末最后一天清仓
        self.stoplost_silent_days = self.params[
            'stoplost_silent_days'] if 'stoplost_silent_days' in self.params else 20  # 止损后不交易的天数
        self.stoplost_level = self.params['stoplost_level'] if 'stoplost_level' in self.params else 0.2  # 止损的下跌幅度（按买入价）

        self.select_list = []
        self.special_select_list = {}
        self.hold_list = []  # 昨收持仓
        self.history_hold_list = []  # 最近持有列表
        self.not_buy_again_list = []  # 最近持有不再购买列表
        self.yestoday_high_limit_list = []  # 昨日涨停列表
        self.stoplost_date = None  # 止损日期，为None是表示未进入止损

        self.utilstool = UtilsToolClass()
        self.utilstool.set_params(name, subportfolio_index)

        self.bought_stocks = {}  # 记录补跌的股票和金额
        self.is_stoplost_or_highlimit = False  # 记录是否卖出过止损的股票

        # 行业列表
        # self.industry_list = []
        # 概念列表
        # self.concept_list = []

        # 设置关仓变量，1/4月不交易
        self.no_trading_today_signal = self.params[
            'no_trading_today_signal'] if 'no_trading_today_signal' in self.params else False

    # 每天准备工作
    def day_prepare(self, context):
        log.info(self.name, '--day_prepare选股前的准备工作函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        subportfolio = context.subportfolios[self.subportfolio_index]

        # 获取昨日持股列表
        self.hold_list = list(subportfolio.long_positions)

        # # 获取最近一段时间持有过的股票列表，放入一个新的列表中
        # self.history_hold_list.append(self.hold_list)
        # # 这个列表只维护最近hold_limit_days天的股票池
        # if len(self.history_hold_list) >= self.hold_limit_days:
        #     self.history_hold_list = self.history_hold_list[-self.hold_limit_days:]
        # temp_set = set()
        # for lists in self.history_hold_list:
        #     for stock in lists:
        #         temp_set.add(stock)
        # # 用于记录最近一段时间内曾经持有的股票，避免重复买入。
        # self.not_buy_again_list = list(temp_set)

        # 获取昨日持股中的涨停列表
        if self.hold_list != []:
            df = get_price(self.hold_list, end_date=context.previous_date, frequency='daily',
                           fields=['close', 'high_limit'], count=1, panel=False, fill_paused=False)
            df = df[df['close'] == df['high_limit']]
            self.yestoday_high_limit_list = list(df.code)
        else:
            self.yestoday_high_limit_list = []

        # 检查空仓期
        self.check_empty_month(context)
        # 检查止损
        self.check_stoplost(context)

    # 基础股票池-全市场选股
    def stockpool(self, context, pool_id=1, index=None, is_filter_kcbj=True, is_filter_st=True, is_filter_paused=True,
                  is_filter_highlimit=True,
                  is_filter_lowlimit=True, is_filter_new=True, is_filter_sold=True, is_updown_limit=True,
                  all_filter=False):
        log.info(self.name, '--stockpool函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        if index is None:
            lists = list(get_all_securities(types=['stock'], date=context.previous_date).index)
        else:
            lists = list(get_index_stocks(index))

        if pool_id == 0:
            pass
        elif pool_id == 1:
            if all_filter:
                lists = self.utilstool.filter_basic_stock(context, lists)
            else:
                if is_filter_kcbj:
                    lists = self.utilstool.filter_kcbj_stock(context, lists)
                if is_filter_st:
                    lists = self.utilstool.filter_st_stock(context, lists, is_updown_limit=is_updown_limit)
                if is_filter_paused:
                    lists = self.utilstool.filter_paused_stock(context, lists)
                if is_filter_highlimit:
                    lists = self.utilstool.filter_highlimit_stock(context, lists)
                if is_filter_lowlimit:
                    lists = self.utilstool.filter_lowlimit_stock(context, lists)
                if is_filter_new:
                    lists = self.utilstool.filter_new_stock(context, lists, days=375)
                if is_filter_sold and self.sold_diff_day > 0:
                    lists = self.utilstool.filter_recently_sold(context, lists, diff_day=self.sold_diff_day)

        return lists

    # 按指数选股票，有未来函数，不建议用
    def stockpool_index(self, context, index, pool_id=1):
        log.info(self.name, '--stockpool_index获取指数成分股函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 获取指数成份股
        lists = list(get_index_stocks(index))
        # ·如果pool_id为0,则直接返回原始的成分股列表。
        if pool_id == 0:
            pass
        # ·如果pool_id为1,则进行进一步的筛选：
        # 。过滤掉创业板（股票代码以'30'开头）、科创板（股票代码以'68'开头）、北交所（股票代码以'8'或'4'开头）的股票。
        # 。过滤掉停牌（paused)、ST(is_st)、当日涨停（day_open等于high_limit)、当日跌停（day_open等于low_limit)的股票。
        # 。过滤掉名称中包含'ST'、"*'、'退'的股票。
        # 返回筛选后的股票列表：将经过筛选的股票列表返回。
        elif pool_id == 1:
            # 过滤创业板、ST、停牌、当日涨停
            # TODO
            current_data = get_current_data()
            # 经过测试，这里可以拿到未来的价格
            # log.error('605179.XSHG', current_data['605179.XSHG'].day_open, '--', current_data['605179.XSHG'].high_limit)
            # log.error('603833.XSHG', current_data['603833.XSHG'].day_open, '--', current_data['603833.XSHG'].high_limit)
            lists = [stock for stock in lists if not (
                    (current_data[stock].day_open == current_data[stock].high_limit) or  # 涨停开盘
                    (current_data[stock].day_open == current_data[stock].low_limit) or  # 跌停开盘
                    current_data[stock].paused or  # 停牌
                    current_data[stock].is_st or  # ST
                    ('ST' in current_data[stock].name) or
                    ('*' in current_data[stock].name) or
                    ('退' in current_data[stock].name) or
                    (stock.startswith('30')) or  # 创业
                    (stock.startswith('68')) or  # 科创
                    (stock.startswith('8')) or  # 北交
                    (stock.startswith('4'))  # 北交
            )
                     ]

        return lists

    # 选股
    def select(self, context):
        log.info(self.name, '--select函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 空仓期控制
        if self.use_empty_month and context.current_dt.month in (self.empty_month):
            self.select_list = ['511880.XSHG']
            return
        # 止损期控制
        if self.use_stoplost and self.stoplost_date is not None:
            self.select_list = ['511880.XSHG']
            return
        self.select_list = []

    # 打印交易计划
    def print_trade_plan(self, context, select_list):
        now = str(context.current_dt.date()) + ' ' + str(context.current_dt.time())
        log.info(self.name, '--print_trade_plan函数--', now)

        # 1.获取子投资组合信息：从context中获取当前的子投资组合subportfolio,以及子投资组合的索引 self.subportfolio_index
        subportfolio = context.subportfolios[self.subportfolio_index]
        positions = subportfolio.long_positions
        positions_count = len(positions)
        current_data = get_current_data()  # 取股票名称

        content = now + ' ' + self.name + " 交易计划：" + "\n"

        # 仓位可用余额
        value_amount = subportfolio.available_cash
        # 遍历当前持仓的股票列表 subportfolio.long_positions,如果某只股票不在选股列表select_list的前self.max_hold_count只股票中，则将其标记为卖出。

        # 实时过滤部分股票，否则也买不了，放出去也没有意义
        target_list = self.utilstool.filter_lowlimit_stock(context, self.select_list)
        target_list = self.utilstool.filter_highlimit_stock(context, target_list)
        target_list = self.utilstool.filter_paused_stock(context, target_list)
        # 股票卖出的条件
        # 1. 有持仓
        # 2. 在目标列表中--不卖
        # 3. 不在目标列表中
        #     涨停：不卖
        #     不涨停：卖
        for stock in positions:
            if stock not in target_list[:self.max_hold_count] and stock not in self.yestoday_high_limit_list:
                last_prices = history(1, unit='1m', field='close', security_list=stock)
                current_data = get_current_data()
                if last_prices[stock][-1] < current_data[stock].high_limit:
                    content = content + stock + ' ' + current_data[stock].name + ' 未涨停卖出-- ' + str(
                        positions[stock].value) + '\n<br> '
                    value_amount = value_amount + positions[stock].value
                    positions_count = positions_count - 1

        # 计算买入金额
        # 如果买入数量buy_count大于0,则将可用现金除以买入数量，得到每只股票的买入金额。
        if len(target_list) > self.max_hold_count:
            buy_count = self.max_hold_count - positions_count
        else:
            buy_count = len(target_list) - positions_count
        if buy_count > 0:
            value_amount = value_amount / buy_count

        # 遍历选股列表
        # 如果某只股票不在当前持仓中，且在选股列表的前 self.max_hold_count只股票中，则将其标记为买入，并添加买入金额
        # 如果某只股票在当前持仓中，且在选股列表的前self.max_hold_count只股票中，则将其标记为继续持有。
        for stock in select_list:
            if stock not in subportfolio.long_positions and stock in select_list[:self.max_hold_count]:
                content = content + stock + ' ' + current_data[
                    stock].name + ' 买入-- ' + str(
                    value_amount) + '\n<br>'
            elif stock in subportfolio.long_positions and stock in select_list[:self.max_hold_count]:
                content = content + stock + ' ' + current_data[stock].name + ' 继续持有 \n<br>'
            else:
                # 兜底逻辑，一般用不到
                content = content + stock + ' ' + current_data[stock].name + '  持仓已满，备选股票 \n<br>'

        if ('买' in content) or ('持有' in content) or ('卖' in content):
            # weixin消息
            send_message(content)
            method_name = inspect.getframeinfo(inspect.currentframe()).function
            item = f"分仓策略:{self.name}<br>-函数名称:{method_name}<br>-时间:{now}"
            self.utilstool.send_wx_message(context, item, content)
            log.info(content)

    ##################################  风控函数群 ##################################

    # 空仓期检查
    def check_empty_month(self, context):
        log.info(self.name, '--check_empty_month函数：空仓期检查--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        subportfolio = context.subportfolios[self.subportfolio_index]
        if self.use_empty_month and context.current_dt.month in (self.empty_month) and len(
                subportfolio.long_positions) > 0:
            content = context.current_dt.date().strftime(
                "%Y-%m-%d") + self.name + ': 进入空仓期' + "\n" + "当前持仓股票: " + "\n"
            for stock in subportfolio.long_positions:
                content = content + stock + "\n"
            log.info(content)

    # 进入空仓期清仓
    def close_for_empty_month(self, context, exempt_stocks=None):
        if exempt_stocks is None:
            exempt_stocks = ['511880.XSHG']

        log.info(self.name, f'--close_for_empty_month函数：在空仓期保留{exempt_stocks}，卖出其他股票--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        subportfolio = context.subportfolios[self.subportfolio_index]
        if self.use_empty_month and context.current_dt.month in self.empty_month and len(
                subportfolio.long_positions) > 0:
            # 获取当前持有的所有股票
            positions = list(subportfolio.long_positions)
            # 排除exempt_stocks中的股票
            stocks_to_sell = [stock for stock in positions if stock not in exempt_stocks]
            if stocks_to_sell:
                self.sell(context, stocks_to_sell)
                log.info(self.name, f'--空仓期卖出股票：{stocks_to_sell}，保留{exempt_stocks}--',
                         str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
            else:
                log.info(self.name, f'--空仓期没有需要卖出的股票，保留{exempt_stocks}--',
                         str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

    # 每月最后一天，清仓等账户均衡
    def close_for_month_last_day(self, context):
        log.info(self.name, '--close_for_month_last_day函数，每月最后一天，清仓等账户均衡--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        subportfolio = context.subportfolios[self.subportfolio_index]
        if self.use_empty_month_last_day and context.current_dt.month in (self.empty_month_last_day) and len(
                subportfolio.long_positions) > 0:
            self.sell(context, list(subportfolio.long_positions))

    # 止损检查
    # 实现了一个止损检查功能，它会根据股票的跌幅来决定是否需要止损，并在需要止损时记录止损日期和打印止损的股票列表。
    def check_stoplost(self, context):
        log.info(self.name, '--check_stoplost函数:止损检查--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        subportfolio = context.subportfolios[self.subportfolio_index]
        if self.use_stoplost:
            if self.stoplost_date is None:
                # 获取持仓股票的当前最新价
                last_prices = history(1, unit='1m', field='close', security_list=subportfolio.long_positions)
                for stock in subportfolio.long_positions:
                    position = subportfolio.long_positions[stock]
                    # 如果股票跌幅超stoplost_level:20%
                    if (position.avg_cost - last_prices[stock][-1]) / position.avg_cost > self.stoplost_level:
                        # 止损日记录到self.stoplost_date中
                        self.stoplost_date = context.current_dt.date()
                        log.info(self.name + ': ' + '开始止损')
                        content = context.current_dt.date().strftime("%Y-%m-%d") + ' ' + self.name + ': 止损' + "\n"
                        for stock in subportfolio.long_positions:
                            content = content + stock + "\n"
                        log.info(content)
                        # 一旦有股票需要止损，就不需要继续检查其他股票了。
                        break
            else:  # 已经在清仓静默期,stoplost_silent_days天后退出静默期
                if (context.current_dt + dt.timedelta(
                        days=-self.stoplost_silent_days)).date() >= self.stoplost_date:
                    self.stoplost_date = None
                    log.info(self.name + ': ' + '退出止损')

    # 止损时清仓
    def close_for_stoplost(self, context, exempt_stocks=None):
        if exempt_stocks is None:
            exempt_stocks = ['511880.XSHG']

        log.info(self.name, f'--close_for_stoplost函数：在止损期保留{exempt_stocks}，卖出其他股票--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        subportfolio = context.subportfolios[self.subportfolio_index]
        if self.use_stoplost and self.stoplost_date is not None and len(subportfolio.long_positions) > 0:
            # 获取当前持有的所有股票
            positions = list(subportfolio.long_positions)
            # 排除exempt_stocks中的股票
            stocks_to_sell = [stock for stock in positions if stock not in exempt_stocks]
            if stocks_to_sell:
                self.sell(context, stocks_to_sell)
                log.info(self.name, f'--止损期卖出股票：{stocks_to_sell}，保留{exempt_stocks}--',
                         str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
            else:
                log.info(self.name, f'--止损期没有需要卖出的股票，保留{exempt_stocks}--',
                         str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

    # 止损检查
    # 实现了一个止损检查功能，它会根据股票的跌幅来决定是否需要止损，并在需要止损时记录止损日期和打印止损的股票列表。
    def stoploss(self, context, stocks_index=None, index_drop_threshold = 0, stock_drop_threshold = 0):
        log.info(self.name, '--stoploss函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        # 定义不同策略对应的指数、跌幅阈值以及个股跌幅阈值
        positions = context.subportfolios[self.subportfolio_index].positions
        # 联合止损：结合大盘及个股情况进行止损判断
        if stocks_index and not (self.use_empty_month and context.current_dt.month in (self.empty_month)):
            # 非1/4再止损
            # 计算指数日内最高和当前价格
            index_data = get_price(stocks_index, start_date=context.current_dt.date(), end_date=context.current_dt,
                                   frequency='1m', fields=['high', 'close'], skip_paused=False, fq='pre', panel=False)
            if not index_data.empty:
                index_high = index_data['high'].max()
                index_current = index_data['close'].iloc[-1]
                index_drop = (index_high - index_current) / index_high
                if index_drop > index_drop_threshold:
                    # 指数下跌超过阈值，清仓对应策略
                    log.info(f"【{self.name}】因{stocks_index}指数下跌超过{index_drop_threshold * 100}%清仓📉")
                    for stock in list(positions.keys()):
                        self.sell(context, [stock])
        else:
            for stock in list(positions.keys()):
                stock_data = get_price(stock, start_date=context.current_dt.date(), end_date=context.current_dt,
                                       frequency='1m', fields=['high', 'close'], skip_paused=False, fq='pre', panel=False)
                if not stock_data.empty:
                    stock_high = stock_data['high'].max()
                    stock_current = stock_data['close'].iloc[-1]
                    stock_drop = (stock_high - stock_current) / stock_high
                    if stock_drop > stock_drop_threshold:
                        # 个股下跌超过阈值，清仓个股并重新调仓
                        if self.sell(context, [stock]):
                            g.global_sold_stock_record[stock] = context.current_dt.date()
                            log.info(f"【{self.name}】{stock} 因下跌超过{stock_drop_threshold * 100}%清仓🚨")
                            self.select(context)
                            self.adjustwithnoRM(context, exempt_stocks=['518880.XSHG'])

    # 3-8 判断今天是否为账户资金再平衡的日期(暂无使用)
    # date_flag,1-单个月，2-两个月1和4，3-三个月1和4和6
    def today_is_between(self, context, date_flag, start_date, end_date):
        today = context.current_dt.strftime('%m-%d')
        # 1(01-01~01-31)-4(04-01~04-30)-6(06-01~06-30)
        if date_flag == 1:
            if (start_date <= today) and (today <= end_date):
                return True
            else:
                return False
        elif date_flag == 2:
            if ('01-01' <= today) and (today <= '01-31'):
                return True
            elif ('04-01' <= today) and (today <= '04-30'):
                return True
            else:
                return False
        elif date_flag == 2:
            if ('01-01' <= today) and (today <= '01-31'):
                return True
            elif ('04-01' <= today) and (today <= '04-30'):
                return True
            elif ('06-01' <= today) and (today <= '06-30'):
                return True
            else:
                return False

    ##################################  交易函数群 ##################################

    # 买入多只股票
    def buy(self, context, buy_stocks, is_single_buy=False):

        log.info(self.name, '--buy函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        subportfolio = context.subportfolios[self.subportfolio_index]
        if is_single_buy and len(subportfolio.long_positions) > 0:
            # 如果有持仓，还有选票就先不买了
            pass

        current_holdings = subportfolio.long_positions
        available_cash = subportfolio.available_cash
        max_hold_count = self.max_hold_count
        current_holding_count = len(current_holdings)

        # 分离buy_stocks为已持仓和未持仓两部分
        held_stocks = [stock for stock in buy_stocks if stock in current_holdings]
        new_stocks = [stock for stock in buy_stocks if stock not in current_holdings]

        # 计算可以买入的未持仓股票数量
        total_new = min(max_hold_count - current_holding_count, len(new_stocks))
        total_held = len(held_stocks)
        log.info(self.buy_strategy_mode, '策略详情:目标股票列表--', buy_stocks,
                 '--最大持仓股票数--', max_hold_count,
                 '--当前持仓股票数--', current_holding_count,
                 '--当前持仓股票明细--', current_holdings,
                 '--目标股票中未持仓股票列表--', new_stocks,
                 '--目标股票中已持仓股票列表--', held_stocks
                 )

        log.info(self.buy_strategy_mode, '策略详情:当前持仓--', current_holdings, '--已持仓股票列表--', held_stocks,
                 '--未持仓股票列表--', new_stocks)

        if self.buy_strategy_mode == 'equal':
            # Strategy 1: Buy new and held stocks equally
            # 计算总的购买金额
            total_value = available_cash
            if (total_new + total_held) <= 0 or total_value <= 0:
                log.info('没有可购买的股票。')
                return

            stock_value = total_value / (total_new + total_held)
            log.debug('equal买入策略：计算总的购买金额：', total_value)
            log.debug('equal买入策略：每只股票的购买金额比例：', stock_value)
            log.debug('equal买入策略：计算可以买入的未持仓股票数量：', total_new, '--待买入列表:', new_stocks)
            log.debug('equal买入策略：计算可以买入的已持仓股票数量：', total_held, '--已持仓列表:', held_stocks)

            # 加仓已持有的股票
            if total_held > 0:
                for stock in held_stocks:
                    if available_cash <= 0:
                        break
                    value = min(stock_value, available_cash)
                    if self.utilstool.open_position(context, stock, value, False):
                        available_cash -= value
                        log.info(f'加仓已持有股票 {stock}，金额: {value}')
                    else:
                        log.warning(f'加仓已持有股票 {stock} 失败，跳过。')

            # 购买新股票
            if total_new > 0:
                for stock in new_stocks:
                    if available_cash <= 0:
                        break
                    value = min(stock_value, available_cash)
                    if self.utilstool.open_position(context, stock, value, False):
                        available_cash -= value
                        log.info(f'买入新股票 {stock}，金额: {value}')
                    else:
                        log.warning(f'买入新股票 {stock} 失败，跳过。')


        elif self.buy_strategy_mode == 'priority':
            # Strategy 2: Prioritize new stocks, then held stocks
            if total_new > 0:
                stock_value = available_cash / total_new
                log.debug('priority买入策略：计算总的购买金额：', available_cash)
                log.debug('priority买入策略：每只股票的购买金额比例：', stock_value)
                log.debug('priority买入策略：计算可以买入的未持仓股票数量：', total_new, '--待买入列表:', new_stocks)
                for stock in new_stocks:
                    if available_cash <= 0:
                        break
                    value = min(stock_value, available_cash)
                    if self.utilstool.open_position(context, stock, value, False):
                        available_cash -= value
                        log.info(f'买入新股票 {stock}，金额: {value}')
                    else:
                        log.warning(f'买入新股票 {stock} 失败，跳过。')

            if total_held > 0:
                stock_value = available_cash / total_held
                log.debug('priority买入策略：计算总的购买金额：', available_cash)
                log.debug('priority买入策略：每只股票的购买金额比例：', stock_value)
                log.debug('priority买入策略：计算可以买入的已持仓股票数量：', total_held, '--待买入列表:', held_stocks)
                for stock in held_stocks:
                    if available_cash <= 0:
                        break
                    value = min(stock_value, available_cash)
                    if self.utilstool.open_position(context, stock, value, False):
                        available_cash -= value
                        log.info(f'加仓已持有股票 {stock}，金额: {value}')
                    else:
                        log.warning(f'加仓已持有股票 {stock} 失败，跳过。')

        else:
            log.warning('无效的策略模式。')
            return

    # 卖出多只股票
    def sell(self, context, sell_stocks):

        log.info(self.name, '--sell函数--要卖出的股票列表--', sell_stocks,
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        subportfolio = context.subportfolios[self.subportfolio_index]
        for stock in sell_stocks:
            if stock in subportfolio.long_positions:
                self.utilstool.close_position(context, stock, 0)

    # 调仓
    def adjustwithnoRM(self, context, only_buy=False, only_sell=False, together=True, is_single_buy=False,
                       exempt_stocks=None):
        log.info(self.name, '--adjustwithnoRM调仓函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        if exempt_stocks is None:
            exempt_stocks = ['511880.XSHG']

        # 空仓期或者止损期不再进行调仓
        if self.use_empty_month and context.current_dt.month in (self.empty_month):
            log.info('adjustwithnoRM调仓函数不再执行，因为当前月份是空仓期，空仓期月份为：', self.empty_month)
            self.buy(context, exempt_stocks, is_single_buy)
            return
        # 止损期控制
        if self.use_stoplost and self.stoplost_date is not None:
            log.info('adjustwithnoRM调仓函数不再执行，因为当前时刻还处于止损期，止损期从:', self.stoplost_date, '开始')
            self.buy(context, exempt_stocks, is_single_buy)
            return

        # 先卖后买
        hold_list = list(context.subportfolios[self.subportfolio_index].long_positions)
        # 售卖列表：不在select_list前max_hold_count中的股票都要被卖掉
        sell_stocks = []
        # 实时过滤部分股票，否则也买不了，放出去也没有意义
        target_list = self.utilstool.filter_highlimit_stock(context, self.select_list)
        target_list = self.utilstool.filter_paused_stock(context, target_list)
        # target_list = self.utilstool.filter_lowlimit_stock(context, target_list)

        log.info(self.name, '--过滤部分股票后的选股列表:', target_list)
        # 股票卖出的条件
        # 1. 有持仓
        # 2. 在目标列表中--不卖
        # 3. 不在目标列表中
        #     涨停：不卖
        #     不涨停：卖

        for stock in hold_list:
            if stock not in target_list[:self.max_hold_count] and stock not in self.yestoday_high_limit_list:
                last_prices = history(1, unit='1m', field='close', security_list=stock)
                current_data = get_current_data()
                if last_prices[stock][-1] < current_data[stock].high_limit:
                    sell_stocks.append(stock)

        if only_buy:
            self.buy(context, target_list, is_single_buy)
            return
        if only_sell:
            self.sell(context, sell_stocks)
            return
        if together:
            self.sell(context, sell_stocks)
            self.buy(context, target_list, is_single_buy)
            return

    # 调仓+均衡资产
    def adjustwithnoRMBalance(self, context, only_buy=False, only_sell=False, together=True, is_single_buy=False,
                              exempt_stocks=None):
        log.info(self.name, '--adjustwithnoRMBalance调仓函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        if exempt_stocks is None:
            exempt_stocks = ['511880.XSHG']

        # 空仓期或者止损期不再进行调仓
        if self.use_empty_month and context.current_dt.month in (self.empty_month):
            log.info('adjustwithnoRM调仓函数不再执行，因为当前月份是空仓期，空仓期月份为：', self.empty_month)
            self.buy(context, exempt_stocks, is_single_buy)
            return
        # 止损期控制
        if self.use_stoplost and self.stoplost_date is not None:
            log.info('adjustwithnoRM调仓函数不再执行，因为当前时刻还处于止损期，止损期从:', self.stoplost_date, '开始')
            self.buy(context, exempt_stocks, is_single_buy)
            return

        # 先卖后买
        hold_list = list(context.subportfolios[self.subportfolio_index].long_positions)
        # 售卖列表：不在select_list前max_hold_count中的股票都要被卖掉
        sell_stocks = []
        # 实时过滤部分股票，否则也买不了，放出去也没有意义
        target_list = self.utilstool.filter_highlimit_stock(context, self.select_list)
        target_list = self.utilstool.filter_paused_stock(context, target_list)
        # target_list = self.utilstool.filter_lowlimit_stock(context, target_list)

        log.info(self.name, '--过滤部分股票后的选股列表:', target_list)
        # 股票卖出的条件
        # 1. 有持仓
        # 2. 在目标列表中--不卖
        # 3. 不在目标列表中
        #     涨停：不卖
        #     不涨停：卖

        for stock in hold_list:
            if stock not in target_list[:self.max_hold_count] and stock not in self.yestoday_high_limit_list:
                last_prices = history(1, unit='1m', field='close', security_list=stock)
                current_data = get_current_data()
                if last_prices[stock][-1] < current_data[stock].high_limit:
                    sell_stocks.append(stock)

        if only_buy:
            self.buy(context, target_list, is_single_buy)
            return
        if only_sell:
            self.sell(context, sell_stocks)
            return
        if together:
            self.sell(context, sell_stocks)
            self.balance_subportfolios(context)
            self.buy(context, target_list, is_single_buy)
            return

    # 平衡账户间资金
    def balance_subportfolios(self, context):
        log.info(f"{self.name}"
                 f"--仓位计划调整的比例:{g.portfolio_value_proportion[self.subportfolio_index]}"
                 f"--仓位调整前的总金额:{context.subportfolios[self.subportfolio_index].total_value}"
                 f"--仓位调整前的可用金额:{context.subportfolios[self.subportfolio_index].available_cash}"
                 f"--仓位调整前的可取金额:{context.subportfolios[self.subportfolio_index].transferable_cash}"
                 f"--仓位调整前的比例:{context.subportfolios[self.subportfolio_index].total_value / context.portfolio.total_value}"
                 )
        target = (
                g.portfolio_value_proportion[self.subportfolio_index]
                * context.portfolio.total_value
        )
        value = context.subportfolios[self.subportfolio_index].total_value
        # 仓位比例过高调出资金
        cash = context.subportfolios[self.subportfolio_index].transferable_cash  # 当前账户可取资金
        if cash > 0 and target < value:
            amount = min(value - target, cash)
            transfer_cash(
                from_pindex=self.subportfolio_index,
                to_pindex=0,
                cash=amount,
            )
            log.info('第', self.subportfolio_index, '个仓位调整了【', amount, '】元到仓位：0')
            # self.get_net_values(context, amount)

        # 仓位比例过低调入资金
        cash = context.subportfolios[0].transferable_cash  # 0号账户可取资金
        if target > value and cash > 0:
            amount = min(target - value, cash)
            transfer_cash(
                from_pindex=0,
                to_pindex=self.subportfolio_index,
                cash=amount,
            )
            log.info('第0个仓位调整了【', amount, '】元到仓位：', self.subportfolio_index)
            # self.get_net_values(context, -amount)

    # 计算策略复权后净值
    def get_net_values(self, context, amount):
        df = g.strategys_values
        if df.empty:
            return
        column_index = self.subportfolio_index - 1
        # 获取最后一天的索引

        last_day_index = len(df) - 1

        # 获取前一天净值
        last_value = df.iloc[last_day_index, column_index]

        # 获取前一天净值
        last_value = df.iloc[last_day_index, column_index]

        # 计算后复权因子, amount 代表分红金额
        g.after_factor[column_index] *= last_value / (last_value - amount)

    def specialBuy(self, context, total_amount=0, split=1):
        log.info(self.name, '--specialBuy调仓函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        special_select_list = self.special_select_list
        select_list = self.select_list
        # 实时过滤部分股票，否则也买不了，放出去也没有意义
        industry_final_stocks = special_select_list.get('行业', [])
        concept_final_stocks = special_select_list.get('概念', [])
        flag = 0
        if concept_final_stocks:
            target_list = self.utilstool.filter_lowlimit_stock(context, concept_final_stocks)
            target_list = self.utilstool.filter_highlimit_stock(context, target_list)
            target_list = self.utilstool.filter_paused_stock(context, target_list)
            flag = 1
        elif industry_final_stocks:
            target_list = self.utilstool.filter_lowlimit_stock(context, industry_final_stocks)
            target_list = self.utilstool.filter_highlimit_stock(context, target_list)
            target_list = self.utilstool.filter_paused_stock(context, target_list)
            flag = 0.5
        else:
            log.debug('当前股票池:',select_list)
            target_list = self.utilstool.filter_lowlimit_stock(context, select_list)
            log.debug('过滤跌停后的股票池:', target_list)
            target_list = self.utilstool.filter_highlimit_stock(context, target_list)
            log.debug('过滤涨停后的股票池:', target_list)
            target_list = self.utilstool.filter_paused_stock(context, target_list)
            log.debug('过滤停牌后的股票池:', target_list)
            flag = 1

        current_data = get_current_data()
        # 持仓列表
        subportfolios = context.subportfolios[self.subportfolio_index]
        if target_list:
            if total_amount > 0:
                for stock in target_list:
                    self.utilstool.open_position(context, stock, total_amount)
            elif split == 1:
                if subportfolios.long_positions:
                    value = subportfolios.available_cash / len(target_list)
                    for stock in target_list:
                        self.utilstool.open_position(context, stock, value)
                else:
                    value = subportfolios.total_value * 0.5 / len(target_list)
                    for stock in target_list:
                        self.utilstool.open_position(context, stock, value)
            elif split == 2:
                if subportfolios.available_cash / subportfolios.total_value > 0.3:
                    value = subportfolios.available_cash * 0.5 if len(
                        target_list) == 1 else subportfolios.available_cash / len(target_list)
                    for stock in target_list:
                        if subportfolios.available_cash / current_data[stock].last_price > 100:
                            self.utilstool.open_position(context, stock, value)
            elif split == 3:
                hold_list = list(subportfolios.positions)
                num = self.max_hold_count - len(hold_list)
                # if (subportfolios.available_cash / subportfolios.total_value > 0.3) and (num > 0):
                value = subportfolios.available_cash * flag / num
                # target_list = [x for x in target_list if x not in hold_list][:num]
                # log.debug(f'过滤前{num}的股票池:{target_list}')
                # log.debug('最终的股票池:', target_list)
                for stock in target_list:
                    if subportfolios.available_cash / current_data[stock].last_price > 100:
                        self.utilstool.open_position(context, stock, value)
            else:
                if subportfolios.available_cash / subportfolios.total_value > 0.3:
                    value = subportfolios.available_cash * flag / len(target_list)
                    for stock in target_list:
                        if subportfolios.available_cash / current_data[stock].last_price > 100:
                            self.utilstool.open_position(context, stock, value)

    def specialSell(self, context, eveny_bar=False, is_st_sell=False):
        log.info(self.name, '--SpecialSell调仓函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 持仓列表
        hold_positions = context.subportfolios[self.subportfolio_index].long_positions
        hold_list = list(hold_positions)
        # 售卖列表：不在select_list前max_hold_count中的股票都要被卖掉
        sell_stocks = []
        date = self.utilstool.transform_date(context, context.previous_date, 'str')
        current_data = get_current_data()  #

        if is_st_sell:
            for stock in hold_list:
                position = hold_positions[stock]

                # 提取关键字段
                df_history = get_price(stock, end_date=context.previous_date, frequency='daily',
                                       fields=['close', 'high_limit'],
                                       count=1, panel=False)

                avg_cost = position.avg_cost
                current_price = position.price
                last_price = current_data[stock].last_price
                high_limit = current_data[stock].high_limit
                low_limit = current_data[stock].low_limit

                # 计算必要指标
                ret = (current_price / avg_cost - 1) * 100 if avg_cost != 0 else -float('inf')

                # 定义核心条件
                cond1 = (last_price != high_limit)  # 今日未触涨停
                cond2_1 = ret < -3  # 亏损超3%
                cond2_1_1 = ret < -5  # 亏损超5%
                cond2_2 = ret > 0  # 盈利
                cond2_3 = (df_history['close'].iloc[0] == df_history['high_limit'].iloc[0])  # 昨日涨停

                # 组合条件并过滤
                if str(context.current_dt)[-8:-6] < '13' and cond1 and (cond2_1_1 or cond2_2 or cond2_3):
                    result = ''
                    if cond2_1_1:
                        result += '亏损超5%--'
                    if cond2_2:
                        result += '盈利--'
                    if cond2_3:
                        result += '昨日涨停--'
                    if last_price > low_limit and position.closeable_amount != 0:  # 防止跌停价卖出
                        log.info('因',result,'卖出ST股票', [stock, get_security_info(stock, date).display_name])
                        sell_stocks.append(stock)
                if str(context.current_dt)[-8:-6] >= '13' and cond1 and (cond2_1 or cond2_2 or cond2_3):
                    result = ''
                    if cond2_1:
                        result += '亏损超3%--'
                    if cond2_2:
                        result += '盈利--'
                    if cond2_3:
                        result += '昨日涨停--'
                    if last_price > low_limit and position.closeable_amount != 0:  # 防止跌停价卖出
                        log.info('因',result,'卖出ST股票', [stock, get_security_info(stock, date).display_name])
                        sell_stocks.append(stock)
        elif eveny_bar:
            for stock in hold_list:
                position = hold_positions[stock]
                # 获取昨日收盘价
                prev_close = attribute_history(stock, 1, '1d', fields=['close'], skip_paused=True)['close'][0]
                # 有可卖出的仓位  &  当前股票没有涨停 & 当前的价格大于持仓价（有收益）
                if ((position.closeable_amount != 0) and (
                        current_data[stock].last_price < current_data[stock].high_limit) and
                        (prev_close < position.avg_cost) and# avg_cost当前持仓成本大于昨日的收盘价，说明亏了
                        (current_data[stock].last_price >= position.avg_cost * 1.002) # 赶紧跑
                        ):
                    log.info('以成本价 * 1.002 卖出', [stock, get_security_info(stock, date).display_name])
                    sell_stocks.append(stock)
        elif str(context.current_dt)[-8:-6] == '11':
            for stock in hold_list:
                position = hold_positions[stock]
                # 有可卖出的仓位  &  当前股票没有涨停 & 当前的价格大于持仓价（有收益）
                if ((position.closeable_amount != 0) and (
                        current_data[stock].last_price < current_data[stock].high_limit) and (
                        current_data[stock].last_price > 1 * position.avg_cost)):  # avg_cost当前持仓成本
                    log.info('止盈卖出', [stock, get_security_info(stock, date).display_name])
                    sell_stocks.append(stock)
        else:
            for stock in hold_list:
                position = hold_positions[stock]

                close_data2 = attribute_history(stock, 4, '1d', ['close'])
                M4 = close_data2['close'].mean()
                MA5 = (M4 * 4 + current_data[stock].last_price) / 5

                # MA5 = MA(stock, check_date=context.current_dt, timeperiod=5)
                # 有可卖出的仓位  &  当前股票没有涨停 & 当前的价格大于持仓价（有收益）
                if ((position.closeable_amount != 0) and (
                        current_data[stock].last_price < current_data[stock].high_limit) and (
                        current_data[stock].last_price > 1 * position.avg_cost)):  # avg_cost当前持仓成本
                    log.info('止盈卖出', [stock, get_security_info(stock, date).display_name])
                    sell_stocks.append(stock)
                # 有可卖出的仓位  &  跌破5日线止损
                if ((position.closeable_amount != 0) and (current_data[stock].last_price < MA5)):
                    log.info('破五日线止损卖出', [stock, get_security_info(stock, date).display_name])
                    sell_stocks.append(stock)

        self.sell(context, sell_stocks)

    # 换手率计算
    def huanshoulv(self, context, stock, is_avg=False):
        log.info(self.name, '--huanshoulv计算换手率函数--涉及股票:',stock,'--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))
        if is_avg:
            # 计算平均换手率
            start_date = context.current_dt - datetime.timedelta(days=20)
            end_date = context.previous_date
            df_volume = get_price(stock, start_date=start_date, end_date=end_date, frequency='daily', fields=['volume'])
            df_cap = get_valuation(stock, end_date=end_date, fields=['circulating_cap'], count=1)
            circulating_cap = df_cap['circulating_cap'].iloc[0] if not df_cap.empty else 0
            if circulating_cap == 0:
                return 0.0
            df_volume['turnover_ratio'] = df_volume['volume'] / (circulating_cap * 10000)
            return df_volume['turnover_ratio'].mean()
        else:
            # 计算实时换手率
            date_now = context.current_dt
            df_vol = get_price(stock, start_date=date_now.date(), end_date=date_now, frequency='1m', fields=['volume'],
                               skip_paused=False, fq='pre', panel=True, fill_paused=False)
            volume = df_vol['volume'].sum()
            date_pre = context.current_dt - datetime.timedelta(days=1)
            df_circulating_cap = get_valuation(stock, end_date=date_pre, fields=['circulating_cap'], count=1)
            circulating_cap = df_circulating_cap['circulating_cap'][0]
            turnover_ratio = volume / (circulating_cap * 10000)
            return turnover_ratio

    # 换手率卖出
    def sell_when_hsl(self, context):
        log.info(self.name, '--sell_when_hsl换手率卖出股票函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        cd = get_current_data()
        thresh = {'破净策略': (0.001, 0.1), '微盘策略': (0.003, 0.1)}
        if self.name not in thresh.keys():
            return
        shrink, expand = thresh[self.name]
        excluded = {'518880.XSHG', '511880.XSHG'}
        filtered_positions = [s for s in context.subportfolios[self.subportfolio_index].long_positions if
                              s not in excluded]

        for s in filtered_positions:
            if cd[s].last_price >= cd[s].high_limit * 0.997:
                # 涨停跳过
                continue
            rt = self.huanshoulv(context, s, False)
            avg = self.huanshoulv(context, s, True)
            if avg == 0:
                continue
            r = rt / avg
            action, icon = '', ''
            if avg < 0.003:
                action, icon = '缩量', '❄️'
            elif rt > expand and r > 2:
                action, icon = '放量', '🔥'
            if action:
                self.is_stoplost_or_highlimit = True
                g.global_sold_stock_record[s] = context.current_dt.date()
                log.info(
                    f"【{self.name}】{action} {s} {get_security_info(s).display_name} 换手率:{rt:.2%}→均:{avg:.2%} 倍率:{r:.1f}x {icon}")
                self.sell(context, [s])

    # 涨停打开卖出
    def sell_when_highlimit_open(self, context):
        log.info(self.name, '--sell_when_highlimit_open涨停打开卖出股票函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        if self.yestoday_high_limit_list != []:
            for stock in self.yestoday_high_limit_list:
                if stock in context.subportfolios[self.subportfolio_index].long_positions:
                    current_data = get_price(stock, end_date=context.current_dt, frequency='1m',
                                             fields=['close', 'high_limit'],
                                             skip_paused=False, fq='pre', count=1, panel=False, fill_paused=True)
                    if current_data.iloc[0, 0] < current_data.iloc[0, 1]:
                        self.sell(context, [stock])
                        g.global_sold_stock_record[stock] = context.current_dt.date()
                        self.is_stoplost_or_highlimit = True
                        content = context.current_dt.date().strftime(
                            "%Y-%m-%d") + ' ' + self.name + ': {}涨停打开，卖出'.format(stock) + "\n"
                        log.info(content)

    # 计算夏普系数的函数
    def cal_sharpe_ratio(self, returns, rf, type):  # portfolio_daily_returns 是一个包含每日收益的列表
        annual_periods = 250  # 假设一年有250个交易日
        sharpe_ratio = 0
        if (type == 'MEAN'):
            returns = returns - rf / annual_periods  # 计算超额收益
            return_mean = np.mean(returns) * annual_periods  # 简单年化收益率 = 投资组合的平均超额收益率 * 年化期数
            std_annualized = returns.std() * np.sqrt(annual_periods)  # 计算年化标准差
            if std_annualized == 0:  # 计算夏普比率
                sharpe_ratio = 0
            else:
                sharpe_ratio = return_mean / std_annualized
        if (type == 'CAGR'):
            returns = returns - rf / annual_periods  # 计算超额收益
            years = len(returns) / annual_periods  # 投资期数
            total = returns.add(1).prod() - 1  # 计算年化收益率
            return_annualized = (total + 1.0) ** (1.0 / years) - 1  # 年化收益率
            std_annualized = returns.std() * np.sqrt(annual_periods)  # 计算年化标准差
            if std_annualized == 0:  # 计算夏普比率
                sharpe_ratio = 0
            else:
                sharpe_ratio = return_annualized / std_annualized
        return sharpe_ratio

    ## 收盘后运行函数
    def after_market_close(self, context):
        now = str(context.current_dt.date()) + ' ' + str(context.current_dt.time())
        log.info(self.name, '--after_market_close收盘后运行函数--', now)

        subportfolio = context.subportfolios[self.subportfolio_index]

        # 计算当前盈利
        title = self.name + '收益率'
        # subportfolio_startcash=context.portfolio.starting_cash*g.portfolio_value_proportion[self.subportfolio_index]+subportfolio.inout_cash
        # 账户累计出入金
        subportfolio_startcash = subportfolio.inout_cash
        if subportfolio_startcash != 0:
            ret_ratio = round((subportfolio.total_value / subportfolio_startcash - 1), 2)
        else:
            ret_ratio = 0

        kv = {title: ret_ratio}
        record(**kv)
        orders = get_orders()
        trades = get_trades()
        # 创建一个 prettytable 对象,打印当天交易信息
        trade_table = PrettyTable(
            ["策略名称", "代码", "证券名称", "交易方向", "交易时间", "交易数量", "交易价格", "盈亏情况"])
        transaction = 0

        if len(trades) > 0:
            for _trade in trades.values():
                if (self.subportfolio_index == orders[_trade.order_id].pindex):
                    transaction += 1
                    # strategy_index = orders[_trade.order_id].pindex
                    strategy_name = self.name
                    security = _trade.security[:20]
                    name = get_security_info(_trade.security).display_name
                    action = '买入' if orders[_trade.order_id].is_buy else '卖出'
                    if orders[_trade.order_id].is_buy == False:
                        # 卖出的时候可以计算收益情况
                        self.trade_num += 1
                        if _trade.price > round(orders[_trade.order_id].avg_cost, 2):
                            # print('交易日志：',name, _trade.price, round(orders[_trade.order_id].avg_cost,2))
                            self.win_num += 1
                        self.win_lose_rate = self.win_num / self.trade_num
                    # print(self.trade_num,self.win_num,self.win_lose_rate)
                    tradedate = _trade.time
                    tradeamount = _trade.amount
                    tradeprice = _trade.price
                    profit_percent_trade = (_trade.price / orders[_trade.order_id].avg_cost - 1) * 100
                    trade_table.add_row(
                        [strategy_name, security, name, action, tradedate, tradeamount, f"{tradeprice:.3f}",
                         f"{profit_percent_trade:.3f}%"])

        method_name = inspect.getframeinfo(inspect.currentframe()).function
        item = f"分仓策略:{self.name}<br>-函数名称:{method_name}<br>-时间:{now}"
        content_log = ''
        content_wx = ''
        if transaction > 0:
            content_wx = content_wx + '#############<br><br><br>' + f"{self.name} 策略当日交易信息: <br>{self.utilstool.pretty_table_to_kv_string(trade_table)}<br>"
            content_log = content_log + '#############\n\n\n' + f"{self.name} 策略当日交易信息: \n{trade_table}\n"

            # write_file(g.logfile,f'\n{trade_table}', append=True)
            # pass
        else:
            content_log = content_log + '#############' + self.name + '当天没有任何交易#############\n'
            content_wx = content_wx + '#############' + self.name + '当天没有任何交易#############<br>'

            # write_file(g.logfile,'-'*20+self.name+'当天没有任何交易'+'-'*20+'\n', append=True)
            # pass

        # 创建一个 prettytable 对象,打印当天持仓信息
        pos_table = PrettyTable(
            ["策略名称", "代码", "证券名称", "买入日期", "买入价格", "现价", "收益率", "持股数", "市值"])
        if len(list(subportfolio.long_positions)) > 0:
            for stock in list(subportfolio.long_positions):
                position = subportfolio.long_positions[stock]
                security = position.security[:20]
                name = get_security_info(position.security).display_name
                buyindate = position.init_time.date()
                buyinprice = position.avg_cost
                currprice = position.price
                # 股票收益率
                profit_percent_hold = (position.price / position.avg_cost - 1) * 100
                # 股票价值
                value = position.value / 10000
                # 股票持股数
                amount = position.total_amount
                pos_table.add_row([self.name, security, name, buyindate, f"{buyinprice:.3f}", f"{currprice:.3f}",
                                   f"{profit_percent_hold:.3f}%", amount, f"{value:.3f}万"])
            # print(f'\n{pos_table}')

            content_wx = content_wx + "#############<br><br><br>" + f"{self.name} 策略当日持仓信息: <br>{self.utilstool.pretty_table_to_kv_string(pos_table)}<br>"
            content_log = content_log + "#############\n\n\n" + f"{self.name} 策略当日持仓信息: \n{pos_table}\n"

            # write_file(g.logfile,f'\n{pos_table}', append=True)
        else:
            content_wx = content_log + '#############' + self.name + '当天没有持仓#############<br>'
            content_log = content_log + '#############' + self.name + '当天没有持仓#############\n'

            # write_file(g.logfile,'-'*20+self.name+'当天没有任何交易'+'-'*20+'\n', append=True)
            # pass

        # 创建一个 prettytable 对象,打印当天账户信息
        account_table = PrettyTable(
            ["日期", "策略名称", "策略总资产", "策略持仓总市值", "策略可用现金", "策略当天出入金", "策略当天收益率",
             "策略累计收益率", "策略胜率", "策略夏普比率", "策略最大回撤", "最大回撤区间"])
        date = str(context.current_dt.date()) + ' ' + str(context.current_dt.time())
        # 账户可用现金
        cash = subportfolio.available_cash / 10000
        # 账户持仓价值
        pos_value = subportfolio.positions_value / 10000
        total_assets = subportfolio.total_value / 10000
        new_data = {'date': date, 'total_value': subportfolio.total_value}
        self.portfolio_value = self.portfolio_value.append(new_data, ignore_index=True)
        # 计算当日之前的资金曲线最高点
        self.portfolio_value['max2here'] = self.portfolio_value['total_value'].expanding().max()
        # 计算历史最高值到当日的剩余量drawdown
        self.portfolio_value['dd2here'] = self.portfolio_value['total_value'] / self.portfolio_value['max2here']
        # 计算回撤完之后剩余量的最小值(也就是最大回撤的剩余量)，以及最大回撤的结束时间
        end_date, remains = tuple(self.portfolio_value.sort_values(by=['dd2here']).iloc[0][['date', 'dd2here']])
        # 计算最大回撤开始时间
        start_date = self.portfolio_value[self.portfolio_value['date'] <= end_date].sort_values(by='total_value',
                                                                                                ascending=False).iloc[
            0]['date']
        max_draw_down = (1 - remains) * 100
        daily_returns = self.portfolio_value['total_value'].pct_change()

        if (self.inout_cash != 0):
            daily_returns.iloc[-1] = (self.portfolio_value['total_value'].iloc[-1] - self.inout_cash) / \
                                     self.portfolio_value['total_value'].iloc[-2] - 1

        self.sharp = self.cal_sharpe_ratio(daily_returns, rf=0.04, type='CAGR')
        if subportfolio_startcash != 0:
            total_return = subportfolio.total_value / subportfolio_startcash - 1
        else:
            total_return = 0
        account_table.add_row([date, self.name, f"{total_assets:.3f}万", f"{pos_value:.3f}万", f"{cash:.3f}万",
                               f"{self.inout_cash / 10000:.3f}万", f"{daily_returns.iloc[-1] * 100:.3f}%",
                               f"{total_return * 100:.3f}%", f"{self.win_lose_rate:.3f}", f"{self.sharp:.3f}",
                               f"{max_draw_down:.3f}%", f"{start_date}到{end_date}"])
        self.previous_portfolio_value = subportfolio.total_value

        content_wx = content_wx + "#############<br><br><br>" + f"{self.name} 策略当日账户信息: <br>{self.utilstool.pretty_table_to_kv_string(account_table)}<br>"
        content_log = content_log + "#############\n\n\n" + f"{self.name} 策略当日账户信息: \n{account_table}\n"

        # write_file(g.logfile,f'\n{account_table}', append=True)

        log.info(content_log)
        self.utilstool.send_wx_message(context, item, content_wx)
        log.info('-------------分割线-------------')
        # write_file(g.logfile,'-'*20+date+'日志终结'+'-'*20+'\n'+'\n', append=True)
        self.inout_cash = 0

    def clear_append_buy_dict(self, context):  # 卖出补跌的仓位
        now = str(context.current_dt.date()) + ' ' + str(context.current_dt.time())
        log.info(self.name, '--clear_append_buy_dict函数--', now)

        if self.bought_stocks:
            for stock, amount in self.bought_stocks.items():
                positions = context.subportfolios[self.subportfolio_index].long_positions
                if stock in positions:
                    self.utilstool.close_position(context, stock, -amount, False)
                # 清空记录
            self.bought_stocks.clear()

    def append_buy_dict(self, context):
        now = str(context.current_dt.date()) + ' ' + str(context.current_dt.time())
        log.info(self.name, '--append_buy_dict 补买函数--', now)
        subportfolios = context.subportfolios[self.subportfolio_index]
        positions = subportfolios.long_positions

        append_buy_dict = {}
        for stock in self.hold_list:
            if stock in positions:
                position = positions[stock]
                current_price = position.price
                avg_cost = position.avg_cost

                if current_price < avg_cost * 0.92:
                    log.info("止损 Selling out %s" % (stock))
                    self.sell(context, [stock])
                    self.is_stoplost_or_highlimit = True
                else:
                    rate = (current_price - avg_cost) / avg_cost
                    append_buy_dict[stock] = rate
        if self.is_stoplost_or_highlimit and append_buy_dict:
            self.is_stoplost_or_highlimit = False
            # 清空记录
            num = 3
            sorted_items = sorted(append_buy_dict.items(), key=lambda x: x[1])  # 按照值进行排序，返回包含(key, value)元组的列表
            result_stock = [item[0] for item in sorted_items[:num]]  # 取前N个元组中的key

            cash = subportfolios.available_cash / num
            log.info("补跌最多的3支 股票代码: %s" % result_stock)
            for stock in result_stock:
                self.utilstool.open_position(context, stock, cash, False)
                if stock not in self.bought_stocks:
                    self.bought_stocks[stock] = cash

class RZQ_Strategy_V3(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)
        self.n_days_limit_up_list = []

    def select(self, context):
        log.info(self.name, '--select函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 根据市场温度设置选股条件，选出股票
        self.select_list = self.__get_rank(context)
        # 编写操作计划
        self.print_trade_plan(context, self.select_list)

    def __get_rank(self, context):
        log.info(self.name, '--get_rank函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        yes_first_no_hl_list = context.yes_first_no_hl_list
        log.info(self.name, '的选股底池--yes_first_no_hl_list:', yes_first_no_hl_list)
        rzq_stocks = []
        current_data = get_current_data()
        date_now = context.current_dt.strftime("%Y-%m-%d")
        mid_time1 = ' 09:15:00'
        end_times1 = ' 09:26:00'
        start = date_now + mid_time1
        end = date_now + end_times1

        # 弱转强
        for s in yes_first_no_hl_list:
            all_date = attribute_history(s, 4, '1d', fields=['close', 'volume', 'money', 'open'],
                                         skip_paused=True)

            # 过滤前面三天涨幅超过28%的票
            if len(all_date) < 4 or (all_date['close'][-1] - all_date['close'][0]) / all_date['close'][0] > 0.28:
                # log.debug('过滤前面三天涨幅超过28%的票')
                continue

            # 过滤前一日收盘价小于开盘价5%以上的票
            open_close_ratio = (all_date['close'][-1] - all_date['open'][-1]) / all_date['open'][-1]
            if open_close_ratio < -0.05:
                # log.debug('过滤前一日收盘价小于开盘价5%以上的票')
                continue

            # 条件：均价，金额，市值，换手率 收盘获利比例低于4%，成交额小于3亿或者大于19亿，或市值小于70亿，大于520亿，过滤
            avg_price_increase_value = all_date['money'][-1] / all_date['volume'][-1] / all_date['close'][-1] - 1
            if avg_price_increase_value < -0.04 or all_date['money'][-1] < 3e8 or all_date['money'][-1] > 19e8:
                # log.debug('均价，金额，市值，换手率')
                continue
            turnover_ratio_data = get_valuation(s, start_date=context.previous_date, end_date=context.previous_date,
                                                fields=['turnover_ratio', 'market_cap', 'circulating_market_cap'])
            if turnover_ratio_data.empty or turnover_ratio_data['market_cap'][0] < 70 or \
                    turnover_ratio_data['circulating_market_cap'][0] > 520:
                # log.debug('均价，金额，市值，换手率2')
                continue


            # 条件：左压
            if self.utilstool.rise_low_volume(context, s):
                continue

            # 条件：高开,开比
            auction_data = get_call_auction(s, start_date=start, end_date=end, fields=['time', 'volume', 'current'])
            if auction_data.empty or auction_data['volume'][0] / all_date['volume'][-1] < 0.03:
                # log.debug('高开,开比1')
                continue
            current_ratio = auction_data['current'][0] / (current_data[s].high_limit / 1.1)
            if current_ratio <= 0.98 or current_ratio >= 1.09:
                # log.debug('高开,开比2')
                continue

            rzq_stocks.append(s)

        log.info('今日弱转强选股：' + ','.join('%s%s' % (s, get_security_info(s).display_name) for s in rzq_stocks))

        return rzq_stocks



class SBGK_Strategy_V3(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)
        self.n_days_limit_up_list = []

    def select(self, context):
        log.info(self.name, '--select函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 根据市场温度设置选股条件，选出股票
        self.select_list = self.__get_rank(context)
        # 编写操作计划
        self.print_trade_plan(context, self.select_list)

    def __get_rank(self, context):
        log.info(self.name, '--get_rank函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        yes_first_hl_list = context.yes_first_hl_list
        current_data = get_current_data()
        log.info(self.name, '的选股底池--yes_first_hl_list:', yes_first_hl_list)
        sbgk_stocks = []
        date_now = context.current_dt.strftime("%Y-%m-%d")
        mid_time1 = ' 09:15:00'
        end_times1 = ' 09:26:00'
        start = date_now + mid_time1
        end = date_now + end_times1

        # 首板高开/低开
        for s in yes_first_hl_list:
            all_date = attribute_history(s, 4, '1d', fields=['close', 'volume', 'money'], skip_paused=True)
            # 获取前一日数据
            # 条件一：均价，金额，市值，换手率 收盘获利比例低于7%，成交额小于5.5亿或者大于20亿，或市值小于70亿，大于520亿，过滤
            avg_price_increase_value = all_date['money'][-1] / all_date['volume'][-1] / all_date['close'][-1] * 1.1 - 1
            if avg_price_increase_value < 0.07 or all_date['money'][-1] < 5.5e8 or all_date['money'][-1] > 20e8:
                continue

            # market_cap 总市值(亿元) > 70亿 流通市值(亿元) < 520亿
            turnover_ratio_data = get_valuation(s, start_date=context.previous_date, end_date=context.previous_date,
                                                fields=['turnover_ratio', 'market_cap', 'circulating_market_cap'])
            # 合并条件一剩余的市值等判断，简化空值和范围判断写法
            if turnover_ratio_data.empty or not (70 <= turnover_ratio_data['market_cap'][0] <= 520):
                continue

            # 条件二：左压
            if self.utilstool.rise_low_volume(context, s):
                continue
            # 条件三：高开,开比
            auction_data = get_call_auction(s, start_date=start, end_date=end, fields=['time', 'volume', 'current'])
            if auction_data.empty or auction_data['volume'][0] / all_date['volume'][-1] < 0.03:
                continue
            current_ratio = auction_data['current'][0] / (current_data[s].high_limit / 1.1)
            if current_ratio <= 1 or current_ratio >= 1.06:
                continue

            # 如果股票满足所有条件，则添加到列表中
            sbgk_stocks.append(s)

        log.info('今日首板高开选股：' + ','.join('%s%s' % (s, get_security_info(s).display_name) for s in sbgk_stocks))

        return sbgk_stocks



class SBDK_Strategy_V3(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)
        self.n_days_limit_up_list = []

    def select(self, context):
        log.info(self.name, '--select函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 根据市场温度设置选股条件，选出股票
        self.select_list = self.__get_rank(context)
        # 编写操作计划
        self.print_trade_plan(context, self.select_list)

    def __get_rank(self, context):
        log.info(self.name, '--get_rank函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        yes_no_first_hl_list = context.yes_no_first_hl_list
        current_data = get_current_data()
        log.info(self.name, '的选股底池--yes_no_first_hl_list:', yes_no_first_hl_list)
        sbdk_stocks = []
        date_now = context.current_dt.strftime("%Y-%m-%d")
        mid_time1 = ' 09:15:00'
        end_times1 = ' 09:26:00'
        start = date_now + mid_time1
        end = date_now + end_times1

        # 首板低开
        if yes_no_first_hl_list:
            date = self.utilstool.transform_date(context, context.previous_date, 'str')
            # 计算相对位置
            rpd = self.utilstool.get_relative_position_df(context, yes_no_first_hl_list, date, 120)
            rpd = rpd[rpd['rp'] <= 0.4]
            stock_list = list(rpd.index)

            # 低开
            df = get_price(stock_list, end_date=date, frequency='daily', fields=['close'], count=1, panel=False,
                           fill_paused=False, skip_paused=True).set_index('code') if len(
                stock_list) != 0 else pd.DataFrame()
            df['open_pct'] = [current_data[s].day_open / df.loc[s, 'close'] for s in stock_list]
            df = df[(0.955 <= df['open_pct']) & (df['open_pct'] <= 0.97)]  # 低开越多风险越大，选择3个多点即可
            stock_list = list(df.index)

            for s in stock_list:
                prev_day_data = attribute_history(s, 1, '1d', fields=['close', 'volume', 'money'], skip_paused=True)
                if prev_day_data['money'][0] >= 1e8:
                    sbdk_stocks.append(s)

        log.info('今日首板低开选股：' + ','.join('%s%s' % (s, get_security_info(s).display_name) for s in sbdk_stocks))

        return sbdk_stocks


class OGT_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)
        self.n_days_limit_up_list = []

    def select(self, context):
        log.info(self.name, '--select函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 根据市场温度设置选股条件，选出股票
        self.select_list = self.__get_rank(context)
        # 编写操作计划
        self.print_trade_plan(context, self.select_list)

    def __get_rank(self, context):
        log.info(self.name, '--get_rank函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        two_hl_list = context.two_hl_list
        three_hl_list = context.three_hl_list
        current_data = get_current_data()
        log.info(self.name, '的选股底池--two_hl_list:', two_hl_list)
        log.info(self.name, '的选股底池--three_hl_list:', three_hl_list)
        ogt_stocks = []
        date_now = context.current_dt.strftime("%Y-%m-%d")
        mid_time1 = ' 09:15:00'
        end_times1 = ' 09:26:00'
        start = date_now + mid_time1
        end = date_now + end_times1

        # 首板高开/低开
        for s in two_hl_list:
            # 条件一：均价，金额，市值，换手率
            prev_day_data = attribute_history(s, 1, '1d', fields=['close', 'volume', 'money'], skip_paused=True)
            avg_price_increase_value = prev_day_data['money'][0] / prev_day_data['volume'][0] / prev_day_data['close'][
                0] * 1.1 - 1
            # 如果平均价格涨幅小于0.07或者前一个交易日的成交金额小于7亿或者大于20亿，则跳过
            if avg_price_increase_value < 0.07 or prev_day_data['money'][0] < 7e8 or prev_day_data['money'][0] > 30e8:
                continue

            # 条件二: market_cap 总市值(亿元) > 70亿, 流通市值(亿元) < 300亿
            turnover_ratio_data = get_valuation(s, start_date=context.previous_date, end_date=context.previous_date,
                                                fields=['turnover_ratio', 'market_cap', 'circulating_market_cap'])
            if turnover_ratio_data.empty or not (70 <= turnover_ratio_data['market_cap'][0] <= 520) or \
                    turnover_ratio_data['circulating_market_cap'][0] > 300:
                continue

            yesterday_turnover_ratio = turnover_ratio_data['turnover_ratio'][0]
            if yesterday_turnover_ratio < 10 or yesterday_turnover_ratio > 30:
                continue

            # 条件三：昨日涨停的成交量为近100日的最大成交量
            # 获取昨日成交量
            yesterday_volume = prev_day_data['volume'][0]

            # 获取过去100个交易日的成交量
            past_volume_data = attribute_history(s, 100, '1d', fields=['volume'], skip_paused=True)
            if past_volume_data.empty:
                continue
            max_past_volume = past_volume_data['volume'].max()
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

            df = get_price(s, end_date=context.previous_date, frequency='daily', fields=['low', 'close', 'low_limit'],
                           count=10,
                           panel=False,
                           fill_paused=False, skip_paused=False)
            low_limit_count = len(df[df.close == df.low_limit])
            if low_limit_count >= 1:
                continue

            # 将符合条件的股票添加到保存的股票列表中
            ogt_stocks.append(s)
        #
        # for s in three_hl_list:
        #     # 过滤前面三天涨幅超过18%的票
        #     price_data = attribute_history(s, 4, '1d', fields=['close'], skip_paused=True)
        #     increase_ratio = (price_data['close'][-1] - price_data['close'][0]) / price_data['close'][0]
        #     if len(price_data) < 4 or increase_ratio > 0.18:
        #         continue
        #
        #     # 条件一：均价，金额，市值，换手率
        #     prev_day_data = attribute_history(s, 1, '1d', fields=['close', 'volume', 'money'], skip_paused=True)
        #     avg_price_increase_value = prev_day_data['money'][0] / prev_day_data['volume'][0] / prev_day_data['close'][
        #         0] * 1.1 - 1
        #     # 如果平均价格涨幅小于0.07或者前一个交易日的成交金额小于7亿或者大于20亿，则跳过
        #     if avg_price_increase_value < 0.07 or prev_day_data['money'][0] < 7e8 or prev_day_data['money'][0] > 20e8:
        #         continue
        #     # 如果换手率为空或者市值小于70，则跳过
        #     turnover_ratio_data = get_valuation(s, start_date=context.previous_date, end_date=context.previous_date,
        #                                         fields=['turnover_ratio', 'market_cap', 'circulating_market_cap'])
        #     if turnover_ratio_data.empty or turnover_ratio_data['market_cap'][0] < 70 or \
        #             turnover_ratio_data['circulating_market_cap'][0] > 300:
        #         continue
        #     # 如果近期有跌停，则跳过
        #     df = get_price(s, end_date=context.previous_date, frequency='daily', fields=['low', 'close', 'low_limit'],
        #                    count=10,
        #                    panel=False,
        #                    fill_paused=False, skip_paused=False)
        #     low_limit_count = len(df[df.close == df.low_limit])
        #     if low_limit_count >= 1:
        #         continue
        #
        #     # 条件二：左压
        #     zyts = self.utilstool.calculate_zyts(context, s)
        #     volume_data = attribute_history(s, zyts, '1d', fields=['volume'], skip_paused=True)
        #     if len(volume_data) < 2 or volume_data['volume'][-1] <= max(volume_data['volume'][:-1]) * 0.90:
        #         continue
        #
        #     # 将符合条件的股票添加到保存的股票列表中
        #     ogt_stocks.append(s)

        ogt_stocks = list(set(ogt_stocks))
        log.info('今日一进二选股：' + ','.join('%s%s' % (s, get_security_info(s).display_name) for s in ogt_stocks))

        return ogt_stocks


class ST_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)
        self.n_days_limit_up_list = []

    def select(self, context):
        log.info(self.name, '--select函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 根据市场温度设置选股条件，选出股票
        self.select_list = self.__get_rank(context)
        # 编写操作计划
        self.print_trade_plan(context, self.select_list)

    def __get_rank(self, context):
        log.info(self.name, '--get_rank函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        current_data = get_current_data()
        st_list = []
        init_st_list = self.utilstool.get_st(context)
        # 1 4 12月 国九
        singal = self.today_is_between(context)
        if singal == True:
            print(f'筛选前市面上所有的ST股票个数：{len(init_st_list)}')
            init_st_list = self.GJT_filter_stocks(init_st_list)
            print(f'筛选后市面上所有的符合国九条ST股票个数：{len(init_st_list)}')

        init_st_list = self.st_filter_stocks(context, init_st_list)
        log.debug(f'基础信息过滤后符合条件的ST股票池：{init_st_list}')
        if len(init_st_list) == 0:
            return st_list
        init_st_list = self.st_rzq_list(context, init_st_list)
        log.debug(f'弱转强过滤后符合条件的ST股票池：{init_st_list}')
        if len(init_st_list) == 0:
            return st_list
            # 低开
        df = get_price(init_st_list, end_date=context.previous_date, frequency='daily', fields=['close'], count=1,
                       panel=False,
                       fill_paused=False, skip_paused=True).set_index('code')
        df['open_now'] = [current_data[s].day_open for s in init_st_list]
        df = df[(df['open_now'] / df['close']) < 1.01]  # 低开越多风险越大，选择3个多点即可
        df = df[(df['open_now'] / df['close']) > 0.95]
        st_list = list(df.index)
        if len(st_list) == 0:
            return st_list
        df = get_valuation(st_list, start_date=context.previous_date,
                           end_date=context.previous_date,
                           fields=['turnover_ratio', 'market_cap', 'circulating_market_cap']
                           )
        df = df.sort_values(by='turnover_ratio', ascending=False)
        st_list = list(df.code)

        log.info('今日ST选股：' + ','.join('%s%s' % (s, get_security_info(s).display_name) for s in st_list))

        return st_list

    def today_is_between(self, context):
        today = context.current_dt.strftime('%m-%d')
        if ('01-15' <= today) and (today <= '01-31'):
            return True
        elif ('04-15' <= today) and (today <= '04-31'):
            return True
        elif ('12-15' <= today) and (today <= '12-31'):
            return True
        else:
            return False

    ##国九条筛选##
    def GJT_filter_stocks(self, stocks):
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

    ##技术指标筛选##
    def st_filter_stocks(self, context, stocks):
        yesterday = pd.Timestamp(context.previous_date)  # 关键修改点
        df = get_price(
            stocks,
            count=11,
            frequency='1d',
            fields=['close', 'low', 'volume', 'money'],
            end_date=yesterday,
            panel=False
        ).reset_index()
        # 按股票分组处理
        grouped = df.groupby('code')
        # 计算技术指标
        ma10 = grouped['close'].transform(lambda x: x.rolling(10).mean())  # 10日均线
        prev_low = grouped['low'].shift(1)  # 前一日最低价
        prev_volume = grouped['volume'].shift(1)  # 前一日成交量
        prev_money = grouped['money'].shift(1)  # 前一日成交量
        # 构建筛选条件
        conditions = (
                (df['close'] > prev_low) &  # 多头排列
                (df['close'] > ma10) &  # 10日线上方
                (df['volume'] > prev_volume) &  # 放量
                # (df['money'] >= 10000000 ) &  # 成交量大于3000w
                (df['volume'] < 10 * prev_volume) &  # 成交量未暴增
                (df['close'] > 1)  # 股价>1
        )

        # 精准获取最新交易日数据（双重验证）
        latest_mask = (df['time'] == yesterday) & (df['time'] == df['time'].max())
        latest_data = df[latest_mask].copy()  # 创建独立副本

        # 在最新数据子集上应用条件
        final_mask = conditions[latest_mask]  # 保持索引对齐
        valid_stocks = latest_data.loc[final_mask, 'code'].unique().tolist()

        return valid_stocks

    ##筛选昨日不涨停的股票##
    def st_rzq_list(self, context, initial_list):
        # 文本日期
        date = context.previous_date

        date_2, date_1, date = get_trade_days(end_date=date, count=3)

        # 昨日不涨停
        h1_list = self.utilstool.get_ever_hl_stock3(context, initial_list, date)
        # 前日涨停过滤
        elements_to_remove = self.utilstool.get_hl_stock(context, initial_list, date_1)

        rzq_list = [stock for stock in h1_list if stock in elements_to_remove]

        return rzq_list
