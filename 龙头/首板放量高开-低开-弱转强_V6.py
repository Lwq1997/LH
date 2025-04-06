# 导入函数库
# -*- coding: utf-8 -*-
# 如果你的文件包含中文, 请在文件的第一行使用上面的语句指定你的文件编码
from datetime import datetime

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
from UtilsToolClass import UtilsToolClass
from SBGK_Strategy_V3 import SBGK_Strategy_V3
from RZQ_Strategy_V3 import RZQ_Strategy_V3
from SBDK_Strategy_V3 import SBDK_Strategy_V3
from OGT_Strategy import OGT_Strategy
from ST_Strategy import ST_Strategy

from Strategy import Strategy


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


def after_market_close(context):
    g.strategys['首板高开'].after_market_close(context)
    g.strategys['弱转强'].after_market_close(context)
    g.strategys['首板低开'].after_market_close(context)
    g.strategys['一进二'].after_market_close(context)
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
