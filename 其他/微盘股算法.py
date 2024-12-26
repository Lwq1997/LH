# 克隆自聚宽文章：https://www.joinquant.com/post/51269
# 标题：微盘指数计算与微盘指数的策略与分析,指数12年涨了170倍
# 作者：chenmq

# 导入函数库
from jqdata import *

import pandas as pd
from jqdata import get_all_trade_days, get_trade_days
from tqdm import tqdm
import numpy as np
from jqlib.technical_analysis import *

# 初始化函数，设定基准等等
def initialize(context):
    # 设定沪深300作为基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    # 输出内容到日志 log.info()
    log.info('初始函数开始运行且全局只运行一次')
    # 过滤掉order系列API产生的比error级别低的log
    # log.set_level('order', 'error')

    ### 股票相关设定 ###
    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.0005, open_commission=0.0000, close_commission=0.0001, min_commission=0),
                   type='stock')

    ## 运行函数（reference_security为运行时间的参考标的；传入的标的只做种类区分，因此传入'000300.XSHG'或'510300.XSHG'是一样的）
    # 开盘前运行
    run_daily(market_open, time='09:31', reference_security='000300.XSHG')
    # 开盘时运行


## 开盘时运行函数
def market_open(context):
    all_stocks = get_all_securities('stock', context.previous_date).index.tolist()
    # 获取股票的ST状态数据，并筛选出非ST股票（假设数据格式符合预期）
    get_extras_sate = get_extras('is_st', all_stocks, start_date=context.previous_date, end_date=context.previous_date,
                                 df=True)
    valid_stocks = [stock for stock in get_extras_sate.columns if not get_extras_sate[stock][0]]
    # 查询这些股票的基本面数据（这里简化示例只取部分字段，可按需完善）
    q = query(valuation.code, valuation.market_cap).filter(
        valuation.code.in_(valid_stocks)).order_by(
        valuation.market_cap.asc()).limit(400)
    df_fundamentals = get_fundamentals(q, date=context.previous_date)
    buy_list = list(df_fundamentals.code)
    position = list(context.portfolio.positions)

    sell_list = [stock for stock in position if stock not in buy_list]
    hold_list = [stock for stock in buy_list if stock not in position]

    for stock in sell_list:
        order_target_value(stock, 0)

    value = context.portfolio.cash / len(hold_list)
    for stock in hold_list:
        order_target_value(stock, value)



 def calculate_microcap_index_series(date_list, count):
     """
     计算给定日期范围及天数内的微盘指数序列以及对应交易日的平均涨幅、布林带相关指标（中值、上轨、下轨和std）。按照等权计算逻辑，以前一天算出的指数点数乘以当天等权股票涨跌幅来计算下一天指数值，初始指数为1000点。

     参数:
     date_list (str): 结束日期字符串，格式需符合相关函数要求，例如 '2024-12-06'。
     count (int): 往前追溯的交易天数数量。

     返回:
     pd.DataFrame: 包含日期、对应微盘指数值、平均涨幅、布林带中值、布林带上轨、布林带下轨以及标准差、涨停股成交量、涨幅大于8%的股票数、涨幅大于8%的股票成交量、涨幅大于5%的交易股票个数、涨幅大于5%的交易股票金额、各涨幅区间股票数量占比、各涨幅区间交易金额占比的数据框。
     """
     before_days = get_trade_days(end_date=date_list, count=count)
     result_df = pd.DataFrame(columns=["date", "microcap_index_value", "average_return",
                                        "index_change_multiplier_veg", "index_change_multiplier_sum",
                                        "bollinger_mid", "bollinger_upper", "bollinger_lower", "std_value",
                                        "index_change_multiplier_high_num", "index_change_multiplier_low_num",
                                        "limit_up_money", "up_8_percent_stock_num", "up_8_percent_stock_volume",
                                        "up_5_percent_stock_num", "up_5_percent_stock_money",
                                        "below_0_percent_stock_num_ratio", "below_0_percent_money_ratio",
                                        "zero_to_3_percent_stock_num_ratio", "zero_to_3_percent_money_ratio",
                                        "three_to_5_percent_stock_num_ratio", "three_to_5_percent_money_ratio",
                                        "five_to_8_percent_stock_num_ratio", "five_to_8_percent_money_ratio",
                                        "above_8_percent_stock_num_ratio", "above_8_percent_money_ratio",
                                        "limit_up_stock_num_ratio", "limit_up_money_ratio"])
     index_value = 1000  # 初始指数值设为1000点
     window_size = 20  # 这里定义布林带计算的窗口大小，可根据实际需求调整

     for i in tqdm(before_days):
         # 获取所有股票列表（这里假设是你想要的获取全部股票的方式，可根据实际调整）
         all_stocks = get_all_securities('stock', i).index.tolist()

         # 获取股票的ST状态数据，并筛选出非ST股票（假设数据格式符合预期）
         get_extras_sate = get_extras('is_st', all_stocks, start_date=i, end_date=i, df=True)
         valid_stocks = [stock for stock in get_extras_sate.columns if not get_extras_sate[stock][0]]

         # 查询这些股票的基本面数据（这里简化示例只取部分字段，可按需完善）
         all_trading_date = list(get_all_trade_days())
         next_day = all_trading_date[all_trading_date.index(i) - 1]
         q = query(valuation.code, valuation.market_cap).filter(
             valuation.code.in_(valid_stocks)).order_by(
             valuation.market_cap.asc()).limit(400)
         df_fundamentals = get_fundamentals(q, date=next_day)

         # 获取股票的价格数据（假设获取的数据格式符合后续处理要求）
         df_prices = get_price(list(df_fundamentals['code']), end_date=i, frequency='daily',
                               fields=['close', 'money', 'high_limit', 'low_limit', 'volume'], count=2, panel=False, fill_paused=False)

         # 分组计算每只股票相对于前一天的涨跌幅，这里假设数据中包含了昨天和前天的价格数据
         grouped = df_prices.groupby('code')
         def calculate_return(group):
             group['return'] = (group['close'] / group['close'].shift(1) - 1).fillna(0)
             return group
         df_prices = grouped.apply(calculate_return).reset_index(drop=True)
         df_prices = df_prices[df_prices["time"] == i]
         df_prices['high_num'] = np.where(df_prices['close'] == df_prices['high_limit'], 1, 0)
         df_prices['low_num'] = np.where(df_prices['close'] == df_prices['low_limit'], 1, 0)

         # 统计涨停股的成交量
         limit_up_stocks = df_prices[df_prices['close'] == df_prices['high_limit']]
         limit_up_money = (limit_up_stocks['money'] / 10000).sum()

         # 统计涨幅大于8%的股票数量及它们的成交量
         up_8_percent_stocks = df_prices[df_prices['return'] > 0.08]
         up_8_percent_stock_num = len(up_8_percent_stocks)
         up_8_percent_stock_volume = up_8_percent_stocks['money'].sum()

         # 统计涨幅大于5%的股票个数及交易金额
         up_5_percent_stocks = df_prices[df_prices['return'] > 0.05]
         up_5_percent_stock_num = len(up_5_percent_stocks)
         up_5_percent_stock_money = (up_5_percent_stocks['money'] / 10000).sum()

         # 统计各涨幅区间的股票数量
         below_0_percent_stocks = df_prices[df_prices['return'] < 0]
         below_0_percent_stock_num = len(below_0_percent_stocks)
         zero_to_3_percent_stocks = df_prices[(df_prices['return'] >= 0) & (df_prices['return'] <= 0.03)]
         zero_to_3_percent_stock_num = len(zero_to_3_percent_stocks)
         three_to_5_percent_stocks = df_prices[(df_prices['return'] > 0.03) & (df_prices['return'] <= 0.05)]
         three_to_5_percent_stock_num = len(three_to_5_percent_stocks)
         five_to_8_percent_stocks = df_prices[(df_prices['return'] > 0.05) & (df_prices['return'] <= 0.08)]
         five_to_8_percent_stock_num = len(five_to_8_percent_stocks)
         limit_up_stock_num = len(limit_up_stocks)

         # 统计各涨幅区间的交易金额
         below_0_percent_money = (below_0_percent_stocks['money'] / 10000).sum()
         zero_to_3_percent_money = (zero_to_3_percent_stocks['money'] / 10000).sum()
         three_to_5_percent_money = (three_to_5_percent_stocks['money'] / 10000).sum()
         five_to_8_percent_money = (five_to_8_percent_stocks['money'] / 10000).sum()
         above_8_percent_money = up_8_percent_stock_volume
         limit_up_money_value = limit_up_money

         # 计算各涨幅区间股票数量占比和交易金额占比
         total_stock_num = len(df_prices)
         below_0_percent_stock_num_ratio = below_0_percent_stock_num / total_stock_num
         zero_to_3_percent_stock_num_ratio = zero_to_3_percent_stock_num / total_stock_num
         three_to_5_percent_stock_num_ratio = three_to_5_percent_stock_num / total_stock_num
         five_to_8_percent_stock_num_ratio = five_to_8_percent_stock_num / total_stock_num
         above_8_percent_stock_num_ratio = up_8_percent_stock_num / total_stock_num
         limit_up_stock_num_ratio = limit_up_stock_num / total_stock_num

         total_money = (df_prices['money'] / 10000).sum()
         below_0_percent_money_ratio = below_0_percent_money / total_money
         zero_to_3_percent_money_ratio = zero_to_3_percent_money / total_money
         three_to_5_percent_money_ratio = three_to_5_percent_money / total_money
         five_to_8_percent_money_ratio = five_to_8_percent_money / total_money
         above_8_percent_money_ratio = above_8_percent_money / total_money
         limit_up_money_ratio = limit_up_money_value / total_money

         # 等权计算，每只股票权重设为1 / 股票数量
         weights = 1 / len(df_prices)
         # 计算指数变化倍数，即所有股票等权涨跌幅之和
         index_change_multiplier = (weights * df_prices['return']).sum()
         index_change_multiplier_veg = df_prices['return'].mean()
         index_change_multiplier_sum = (df_prices['money'] / 10000).sum()
         index_change_multiplier_high_num = df_prices['high_num'].sum()
         index_change_multiplier_low_num = df_prices['low_num'].sum()
         # 计算平均涨幅，即所有股票涨跌幅之和除以股票数量，和指数变化倍数在等权情况下是一样的计算逻辑
         average_return = index_change_multiplier

         # 根据前一天指数点数和当天变化倍数计算当天指数值
         index_value *= (1 + index_change_multiplier)

         # 构建新的数据行字典，保证字段名顺序和数据对应准确
         new_row_data = {
             "date": i,
             "microcap_index_value": index_value,
             "average_return": average_return,
             "index_change_multiplier_veg": index_change_multiplier_veg,
             "index_change_multiplier_sum": index_change_multiplier_sum,
             "bollinger_mid": None,
             "bollinger_upper": None,
             "bollinger_lower": None,
             "std_value": None,
             "index_change_multiplier_high_num": index_change_multiplier_high_num,
             "index_change_multiplier_low_num": index_change_multiplier_low_num,
             "limit_up_money": limit_up_money,
             "up_8_percent_stock_num": up_8_percent_stock_num,
             "up_8_percent_stock_volume": up_8_percent_stock_volume,
             "up_5_percent_stock_num": up_5_percent_stock_num,
             "up_5_percent_stock_money": up_5_percent_stock_money,
             "below_0_percent_stock_num_ratio": below_0_percent_stock_num_ratio,
             "below_0_percent_money_ratio": below_0_percent_money_ratio,
             "zero_to_3_percent_stock_num_ratio": zero_to_3_percent_stock_num_ratio,
             "zero_to_3_percent_money_ratio": zero_to_3_percent_money_ratio,
             "three_to_5_percent_stock_num_ratio": three_to_5_percent_stock_num_ratio,
             "three_to_5_percent_money_ratio": three_to_5_percent_money_ratio,
             "five_to_8_percent_stock_num_ratio": five_to_8_percent_stock_num_ratio,
             "five_to_8_percent_money_ratio": five_to_8_percent_money_ratio,
             "above_8_percent_stock_num_ratio": above_8_percent_stock_num_ratio,
             "above_8_percent_money_ratio": above_8_percent_money_ratio,
             "limit_up_stock_num_ratio": limit_up_stock_num_ratio,
             "limit_up_money_ratio": limit_up_money_ratio
         }

         # 将新行数据转换为DataFrame并与结果数据框合并，调整concat顺序确保列顺序正确
         new_row_df = pd.DataFrame([new_row_data])
         result_df = pd.concat([result_df, new_row_df], ignore_index=True)

         # 计算布林带相关指标，基于包含当日数据的result_df来计算
         index_series = result_df["microcap_index_value"].iloc[-window_size:] if len(result_df) >= window_size else result_df["microcap_index_value"]
         std_value = index_series.std()
         bollinger_mid = index_series.mean()
         bollinger_upper = bollinger_mid + 2 * std_value
         bollinger_lower = bollinger_mid - 2 * std_value

         # 使用loc准确更新当日数据所在行的布林带相关字段
         row_index = result_df[result_df["date"] == i].index[0]
         result_df.loc[row_index, "bollinger_mid"] = bollinger_mid
         result_df.loc[row_index, "bollinger_upper"] = bollinger_upper
         result_df.loc[row_index, "bollinger_lower"] = bollinger_lower
         result_df.loc[row_index, "std_value"] = std_value
         result_df = result_df[["date", "microcap_index_value", "average_return",
                                 "index_change_multiplier_veg", "index_change_multiplier_sum",
                                 "bollinger_mid", "bollinger_upper", "bollinger_lower", "std_value",
                                 "index_change_multiplier_high_num", "index_change_multiplier_low_num",
                                 "limit_up_money", "up_8_percent_stock_num", "up_8_percent_stock_volume",
                                 "up_5_percent_stock_num", "up_5_percent_stock_money",
                                 "below_0_percent_stock_num_ratio", "below_0_percent_money_ratio",
                                 "zero_to_3_percent_stock_num_ratio", "zero_to_3_percent_money_ratio",
                                 "three_to_5_percent_stock_num_ratio", "three_to_5_percent_money_ratio",
                                 "five_to_8_percent_stock_num_ratio", "five_to_8_percent_money_ratio",
                                 "above_8_percent_stock_num_ratio", "above_8_percent_money_ratio",
                                 "limit_up_stock_num_ratio", "limit_up_money_ratio"]]

     return result_df
