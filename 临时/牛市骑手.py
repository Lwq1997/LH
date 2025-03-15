# 克隆自聚宽文章：https://www.joinquant.com/post/54065
# 标题：牛市骑手系列完美适配当下市场！
# 作者：奇点方程式

from jqdata import *
import pandas as pd
from datetime import datetime, timedelta


# 初始化函数，设定策略参数
def initialize(context):
    set_option('use_real_price', True)
    log.set_level('system', 'error')
    set_option('avoid_future_data', True)
    # 设置最大持仓股票数量
    g.max_hold_num = 5
    # 用于存储选股列表
    g.target_list = []


def after_code_changed(context):
    unschedule_all()  # 取消所有定时运行
    run_daily(select_stocks, '15:00')  # 每天收盘后选股
    run_daily(buy, '09:30')  # 每天开盘后买入
    run_daily(sell, '14:50')  # 收盘前卖出


# 获取当天涨停的股票，过滤掉上市不足半年的票
def get_today_limit_stocks(search_date):
    # 获取所有股票，排除上市不足半年
    all_stocks = get_all_securities(types=['stock'], date=search_date)
    # 半年前的日期
    pre_half_year_date = datetime.strptime(search_date, "%Y-%m-%d") - timedelta(days=180)
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
def get_stock_industry_df(search_date, stocks):
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


# 获取指定日期范围内的行业热度
def get_industry_heat(start_date, end_date):
    industry_heat_dict = {}  # 存储行业热度的字典
    date_range = pd.date_range(start=start_date, end=end_date)

    for search_date in date_range:
        search_date = search_date.strftime('%Y-%m-%d')
        # 获取当天涨停的股票
        today_limit_stocks = get_today_limit_stocks(search_date)
        # 获取股票所属行业
        stock_industry_df = get_stock_industry_df(search_date, today_limit_stocks)
        # 统计每个行业的涨停股数量
        stock_industry_df["涨停数量"] = 1
        industry_count_df = stock_industry_df.groupby(["sw_L2"]).count()
        industry_count_df = industry_count_df.drop(["code", "sw_L1", "sw_L3"], axis=1)

        # 累加行业热度
        for industry, count in industry_count_df.iterrows():
            if industry in industry_heat_dict:
                industry_heat_dict[industry] += count["涨停数量"]
            else:
                industry_heat_dict[industry] = count["涨停数量"]

    # 将行业热度字典转换为 DataFrame
    industry_heat_df = pd.DataFrame(list(industry_heat_dict.items()), columns=["行业", "涨停总数"])
    industry_heat_df = industry_heat_df.sort_values(by="涨停总数", ascending=False)

    # 输出结果
    print("行业热度统计（{} 至 {}）：".format(start_date, end_date))
    print(industry_heat_df)

    return industry_heat_df


# 选股函数，筛选出行业热度排名前三的股票
def select_stocks(context, ):
    # 获取最近 5 天的行业热度
    start_date = (context.current_dt.date() - timedelta(days=5)).strftime('%Y-%m-%d')
    end_date = context.current_dt.date().strftime('%Y-%m-%d')
    industry_heat_df = get_industry_heat(start_date, end_date)

    # 获取行业热度排名前三的行业
    top_industries = industry_heat_df.head(3)["行业"].tolist()
    print("行业热度排名前三的行业：", top_industries)

    # 获取所有股票
    stock_list = get_all_securities('stock').index.tolist()
    qualified_stocks = []

    for s in stock_list:
        # 排除科创板和创业板
        if s.startswith('688') or s.startswith('300') or s.startswith('301'):
            continue

        # 过滤ST股、停牌股和次新股
        if filter_st_paused_stock([s], context.current_dt.date()) and filter_new_stock([s], context.current_dt.date()):
            # 获取前20日的历史数据
            history_data = attribute_history(s, 20, '1d', ['close', 'volume'], skip_paused=True)
            if history_data.empty:
                continue

            # 条件1：股价在20日均线上方，且短期均线多头排列
            ma5 = history_data['close'][-5:].mean()
            ma10 = history_data['close'][-10:].mean()
            ma20 = history_data['close'].mean()
            if not (ma5 > ma10 > ma20 and history_data['close'][-1] > ma20):
                continue

            # 条件2：前一日成交量较前两日放大至少1.5倍
            if history_data['volume'][-1] < history_data['volume'][-2] * 1.5:
                continue

            # 条件3：前面10天中有4天成交量温和放大
            if not check_volume_increase(history_data['volume'][-11:-1]):
                continue

            # 条件4：近5日平均成交量比例大于3.5%
            turnover_ratio_data = get_valuation(s, end_date=context.previous_date, count=5, fields=['turnover_ratio'])
            if turnover_ratio_data.empty or turnover_ratio_data['turnover_ratio'].mean() < 3.5:
                continue

            # 条件5：MACD金叉且处于零轴上方
            macd_data = get_macd(s, end_date=context.previous_date, frequency='daily')
            if macd_data.empty or not (macd_data['DIF'][-1] > macd_data['DEA'][-1] and macd_data['DIF'][-1] > 0):
                continue

            # 条件6：KDJ的J值大于50
            kdj_data = get_kdj(s, end_date=context.previous_date, frequency='daily')
            if kdj_data.empty or kdj_data['J'][-1] < 50:
                continue

            # 获取股票所属行业
            stock_industry = get_industry(s, date=context.current_dt.date())
            industry_name = stock_industry[s]["sw_l2"]["industry_name"]

            # 如果股票所属行业在热度排名前三的行业中，则加入选股列表
            if industry_name in top_industries:
                qualified_stocks.append(s)

    # 将选股结果存储到全局变量
    g.target_list = qualified_stocks
    print("今日选股: " + str(g.target_list))


def check_volume_increase(volume_data):
    """
    检查前10天中是否有4天成交量温和放大
    """
    count = 0
    for i in range(1, len(volume_data)):
        ratio = volume_data[i] / volume_data[i - 1]
        if 1.1 <= ratio <= 3:  # 温和放大条件
            count += 1
    return count >= 4  # 满足条件的天数 ≥ 4天


def filter_new_stock(initial_list, date, days=250):
    return [s for s in initial_list if get_security_info(s).start_date < date - timedelta(days=days)]


def filter_st_paused_stock(initial_list, date):
    current_data = get_current_data()
    return [s for s in initial_list if
            not (current_data[s].is_st or current_data[s].paused or '退' in current_data[s].name)]


def get_macd(stock, end_date, frequency='daily'):
    # 获取前26日的数据（排除当日）
    data = get_price(stock, end_date=end_date, frequency=frequency, fields=['close'], count=26, panel=False,
                     fill_paused=False)
    if data.empty:
        return pd.DataFrame()
    close = data['close']
    short_ema = close.ewm(span=12, adjust=False).mean()
    long_ema = close.ewm(span=26, adjust=False).mean()
    dif = short_ema - long_ema
    dea = dif.ewm(span=9, adjust=False).mean()
    macd = pd.DataFrame({'DIF': dif, 'DEA': dea})
    return macd


def get_kdj(stock, end_date, frequency='daily'):
    # 获取前9日的数据（排除当日）
    data = get_price(stock, end_date=end_date, frequency=frequency, fields=['close', 'high', 'low'], count=9,
                     panel=False, fill_paused=False)
    if data.empty:
        return pd.DataFrame()
    high = data['high'].rolling(window=9).max()
    low = data['low'].rolling(window=9).min()
    rsv = (data['close'] - low) / (high - low) * 100
    k = rsv.ewm(com=2).mean()
    d = k.ewm(com=2).mean()
    j = 3 * k - 2 * d
    kdj = pd.DataFrame({'K': k, 'D': d, 'J': j})
    return kdj


# 买入函数
def buy(context):
    # 获取选股列表
    target_list = g.target_list
    if target_list:
        print("今日买入: " + str(target_list))

        # 计算每只股票的资金分配
        cash_per_stock = context.portfolio.available_cash / min(len(target_list), g.max_hold_num)

        # 买入选中的股票
        for s in target_list[:g.max_hold_num]:
            # 如果持仓数量已达上限，则停止买入
            if len(context.portfolio.positions) >= g.max_hold_num:
                break
            # 确保买入金额足够
            if cash_per_stock > context.portfolio.available_cash:
                cash_per_stock = context.portfolio.available_cash
            if cash_per_stock > 0:
                order_value(s, cash_per_stock, MarketOrderStyle())
                print('买入: ' + s)


# 卖出函数
def sell(context):
    current_data = get_current_data()
    positions = list(context.portfolio.positions)

    # 如果持仓股票数量超过最大限制，则卖出涨幅最低的股票
    if len(positions) > g.max_hold_num:
        # 计算每只股票的涨幅
        performance = {}
        for s in positions:
            position = context.portfolio.positions[s]
            avg_cost = position.avg_cost
            current_price = current_data[s].last_price
            performance[s] = (current_price - avg_cost) / avg_cost

        # 按涨幅排序，卖出涨幅最低的股票
        sorted_stocks = sorted(performance.items(), key=lambda x: x[1])
        for s, _ in sorted_stocks[:len(positions) - g.max_hold_num]:
            order_target_value(s, 0)
            print('卖出持仓过多的股票: ' + s)

    # 如果股价跌破4日均线，则卖出
    for s in positions:
        close_data = attribute_history(s, 4, '1d', ['close'])
        ma4 = close_data['close'].mean()
        if current_data[s].last_price < ma4:
            order_target_value(s, 0)
            print('卖出跌破4日均线的股票: ' + s)
