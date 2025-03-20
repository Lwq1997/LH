# 克隆自聚宽文章：https://www.joinquant.com/post/54039
# 标题：妖股探测器
# 作者：jack010

# 风险及免责提示：该策略由聚宽用户在聚宽社区分享，仅供学习交流使用。
# 原文一般包含策略说明，如有疑问请到原文和作者交流讨论。
# 原文网址：https://www.joinquant.com/post/37894
# 标题：实盘一周50%的妖股探测器
# 作者：Dr.QYQ

# 回测设置为 分钟

'''
优化说明:
    1.使用修正标准分
        rsrs_score的算法有：
            仅斜率slope，效果一般；
            仅标准分zscore，效果不错；
            修正标准分 = zscore * r2，效果最佳;
            右偏标准分 = 修正标准分 * slope，效果不错。
    2.将原策略的每次持有两只etf改成只买最优的一个，收益显著提高
    3.将每周调仓换成每日调仓，收益显著提高
    4.因为交易etf，所以手续费设为万分之三，印花税设为零，未设置滑点
    5.修改股票池中候选etf，删除银行，红利等收益较弱品种，增加纳指etf以增加不同国家市场间轮动的可能性
    6.根据研报，默认参数介已设定为最优
    7.加入防未来函数
    8.增加择时与选股模块的打印日志，方便观察每笔操作依据
'''

# 导入函数库
from jqdata import *
from jqlib.technical_analysis import *
import numpy as np
from datetime import date, timedelta


# 初始化函数
def initialize(context):
    # 设定沪深300作为基准
    set_benchmark('000300.XSHG')
    # 用真实价格交易
    set_option('use_real_price', True)
    # 打开防未来函数
    set_option("avoid_future_data", True)
    # 将滑点设置为0
    set_slippage(FixedSlippage(0.001))
    # 设置交易成本万分之三
    # set_order_cost(OrderCost(open_tax=0, close_tax=0, open_commission=0.0003, close_commission=0.0003, close_today_commission=0, min_commission=5),
    #               type='fund')
    # 股票类每笔交易时的手续费是：买入时无佣金，卖出时佣金万分之1.5，印花税0.1%, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.001, close_commission=0.00015, min_commission=5), type='stock')
    # 过滤order中低于error级别的日志
    log.set_level('order', 'error')
    # 初始化各类全局变量

    # 动量轮动参数
    g.stock_num = 1  # 筛选的标的支数。
    g.stock_tobuy = 1  # 需要购买的股票数
    g.momentum_day = 29  # 最新动量参考最近momentum_day的
    g.num_days = 5  # 计算分数变化
    g.run_today_day = 1  # 计算几日前的rank stock
    # rsrs择时参数
    g.ref_stock = '000300.XSHG'  # 用ref_stock做择时计算的基础数据
    g.N = 14  # 计算最新斜率slope，拟合度r2参考最近N天
    g.M = 600  # 计算最新标准分zscore，rsrs_score参考最近M天
    g.score_threshold = 0.7  # rsrs标准分指标阈值
    # 个股择时参数
    g.sec_data_num = 5  # 个股数据点数
    # g.take_profit = 0.12 # 移动止盈
    # ma择时参数
    g.mean_day = 7  # 计算ref_stock结束ma收盘价，参考最近mean_day
    g.mean_diff_day = 10  # 计算初始ma收盘价，参考(mean_day + mean_diff_day)天前，窗口为mean_diff_day的一段时间
    g.slope_series = initial_slope_series()[:-1]  # 除去回测第一天的slope，避免运行时重复加入
    # 设置交易时间，每天运行
    run_daily(my_trade, time='09:31', reference_security='000300.XSHG')
    run_daily(check_lose, time='14:50', reference_security='000300.XSHG')
    # run_daily(check_profit, time='10:00')
    run_daily(print_trade_info, time='15:05', reference_security='000300.XSHG')


# 0-0 选取股票池
def get_stock_pool():
    # preday = str(date.today() - timedelta(1)) # get previous date
    # 从多个热门概念中选出市值在50亿以上,500亿以下的标的。

    concepts1 = {
        #  'SC0010'  : '互联网金融',
        #  'SC0011'  : '网络游戏',
        'SC0013': '在线教育',
        'SC0016': '数据中心',
        'SC0020': '汽车电子',
        'SC0022': '人工智能',
        'SC0023': '无人驾驶',
        #  'SC0027'  : '微信小程序',
        'SC0033': '边缘计算',
        'SC0050': '大数据',
        'SC0052': '国产软件',
        'SC0057': '网络安全',
        'SC0058': '无人机',
        'SC0060': '云计算',
        'SC0061': '国产操作系统',
        'SC0069': '智能医疗',
        'SC0196': '量子通信',
        'SC0217': '高端装备',
        'SC0223': '钠离子电池',
        'SC0226': '华为海思概念',
        'SC0232': '储能',
        'SC0247': '数据安全',
        'SC0255': '汽车芯片',
        'SC0257': '互联网医疗',
        'SC0318': '固态电池',
        'SC0327': '数据要素',
        'SC0329': '英伟达概念',
        'SC0331': '算力租赁',
        #  'SC0332'  : '数据确权',
        'SC0342': 'PEEK材料',
        'SC0343': '小米汽车',
        'SC0345': '飞行汽车(eVTOL)',
        'SC0346': '人形机器人',
        'SC0347': 'AI手机',
        'SC0348': '低空经济',
        'SC0349': '铜缆高速连接',
        'SC0350': 'AI语料',
        'SC0355': 'Sora概念(文生视频)',
        'SC0361': '机器人概念',
        'SC0363': '芯片概念',
        #  'SC0367'  : '抖音概念',
        #  'SC0390'  : '锂电池概念',
        'SC0395': '集成电路概念',
        'SC0398': 'AI眼镜',
        'SC0403': '智谱AI',
        'SC0408': 'AIGC',
        #  'SC0409'  : '6G',
        #  'SC0418'  : 'ChatGPT',
        'SC0423': 'AI智能体',
        'SC0424': 'DeepSeek'
    }

    all_concepts = get_concepts()
    concept_codes = []
    for code in concepts1:
        concept_codes.append(code)

    all_concept_stocks = []

    for concept in concept_codes:
        all_concept_stocks += get_concept_stocks(concept)

    q = query(valuation.code).filter(valuation.market_cap >= 30, valuation.market_cap <= 500,
                                     valuation.code.in_(all_concept_stocks))
    stock_df = get_fundamentals(q)
    stock_pool = [code for code in stock_df['code']]
    # 移除创业板和科创板标的
    stock_pool = [code for code in stock_pool if not (code.startswith('30') or code.startswith('688'))]
    stock_pool = filter_st_stock(stock_pool)  # 去除st
    stock_pool = filter_paused_stock(stock_pool)  # 去除停牌
    return stock_pool


# 1-1 选股模块-动量因子轮动
# 基于股票年化收益和判定系数打分,并按照分数从大到小排名
def get_rank(stock_pool, run_today_day, context):
    pre_date = context.current_dt - timedelta(run_today_day)
    score_list = []
    for stock in stock_pool:
        # data = attribute_history(stock, g.momentum_day, '1d', ['close'])
        data = get_price(stock, end_date=pre_date, count=g.momentum_day, frequency='daily',
                         fields='close', skip_paused=True)
        y = data['log'] = np.log(data.close)
        # print(f'length of y is {len(y)}')
        x = data['num'] = np.arange(data.log.size)
        slope_new = (y[-1] - y[0]) / g.momentum_day  # 最近一天收盘价-第一天收盘价/天数
        intercept = y[0]
        # slope, intercept = np.polyfit(x, y, 1)
        annualized_returns = math.pow(math.exp(slope_new), 250) - 1
        r_squared = 1 - (sum((y - (slope_new * x + intercept)) ** 2) / ((len(y) - 1) * np.var(y, ddof=1)))
        score = annualized_returns * np.abs(r_squared)
        # score = annualized_returns
        score_list.append(score)
    stock_dict = dict(zip(stock_pool, score_list))
    sort_list = sorted(stock_dict.items(), key=lambda item: item[1], reverse=True)  # True为降序
    code_list = []
    score_list_sort = []
    for i in range(g.stock_num):
        code_list.append(sort_list[i][0])
        score_list_sort.append(sort_list[i][1])
    rank_stock = code_list[0:g.stock_num]
    rank_stock_score = score_list_sort[0:g.stock_num]
    code_name = []
    for c in range(g.stock_num):
        code_name.append(get_security_info(code_list[c]).display_name)
    print(f'今日自选股{rank_stock}: {code_name} 评分{rank_stock_score}')
    return rank_stock, rank_stock_score


def rank_stock_change(stock_pool, context):
    '''rank_stock_plot
    line plot the score variation for *num_days* for each rank stock.
    '''
    # stock_pool = get_stock_pool()
    # rank_stock, _ = get_rank(
    #    stock_pool, g.run_today_day)  # get today's rank stock
    # print(f'in rank_stock_change, stock pool is {stock_pool}')
    rank_stock_dict_list = []
    for day in range(g.num_days):
        rank_stock_tmp, rank_stock_score = get_rank(stock_pool, day + 1, context)  # day starts from 0
        # print(f'rank_stock_tmp is {rank_stock_tmp}')
        rank_stock_dict_list.append(
            dict(zip(rank_stock_tmp, rank_stock_score)))

    g.stock_df = pd.DataFrame(rank_stock_dict_list)
    g.stock_df = g.stock_df[::-1].reset_index(drop=True).drop_duplicates()  # 最新的在最下面
    # print(f'stock_df column is: {g.stock_df.columns}')
    return g.stock_df


# 2-1 择时模块-计算线性回归统计值
# 对输入的自变量每日最低价x(series)和因变量每日最高价y(series)建立OLS回归模型,返回元组(截距,斜率,拟合度)
def get_ols(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    r2 = 1 - (sum((y - (slope * x + intercept)) ** 2) / ((len(y) - 1) * np.var(y, ddof=1)))
    return (intercept, slope, r2)


# 2-2 择时模块-设定初始斜率序列
# 通过前M日最高最低价的线性回归计算初始的斜率,返回斜率的列表
def initial_slope_series():
    data = attribute_history(g.ref_stock, g.N + g.M, '1d', ['high', 'low'])
    return [get_ols(data.low[i:i + g.N], data.high[i:i + g.N])[1] for i in range(g.M)]


# 2-3 择时模块-计算标准分
# 通过斜率列表计算并返回截至回测结束日的最新标准分
def get_zscore(slope_series):
    mean = np.mean(slope_series)
    std = np.std(slope_series)
    return (slope_series[-1] - mean) / std


# 2-4 择时模块-计算综合信号
# 1.获得rsrs与MA信号,rsrs信号算法参考优化说明，MA信号为一段时间两个端点的MA数值比较大小
# 2.信号同时为True时返回买入信号，同为False时返回卖出信号，其余情况返回持仓不变信号
# 3.改进：加入个股的卖点判据
def get_timing_signal(stock, context):
    '''
    计算大盘信号: RSRS + MA
    '''
    # 计算MA信号
    close_data = attribute_history(g.ref_stock, g.mean_day + g.mean_diff_day, '1d', ['close'])
    today_MA = close_data.close[g.mean_diff_day:].mean()
    before_MA = close_data.close[:-g.mean_diff_day].mean()
    # 计算rsrs信号
    high_low_data = attribute_history(g.ref_stock, g.N, '1d', ['high', 'low'])
    intercept, slope, r2 = get_ols(high_low_data.low, high_low_data.high)
    g.slope_series.append(slope)
    rsrs_score = get_zscore(g.slope_series[-g.M:]) * r2  # 修正标准分
    print(f'today_MA is {today_MA}, before_MA is {before_MA}, rsrs score is {rsrs_score}')
    '''
    个股择时：
    1. MA5买卖点
    2. 3日斜率
    3. 移动止盈
    4. 效果不如不要。。。。
    5. 试试MCAD
    6. 试试KDJ
    '''
    # 计算个股x日斜率
    close_data_sec = attribute_history(stock, g.sec_data_num, '1d', ['close'])
    current_price = attribute_history(stock, 1, '1m', ['close'])  # get current stock price
    MA_sec = close_data_sec.close.mean()
    print(f'现价:{current_price.close[-1]}, MA{g.sec_data_num}: {MA_sec}')
    # 计算2日斜率
    # close_data_sec_ = attribute_history(stock, 2, '1d', ['close'])
    # y = close_data_sec_['log'] = np.log(close_data_sec_.close)
    # x = close_data_sec_['num'] = np.arange(close_data_sec_.log.size)
    # slope_sec,_ = np.polyfit(x,y,1)
    # print(f'Slope < 0: {slope_sec<0}')
    # 移动止盈
    # stock_data = attribute_history(stock,g.sec_data_num,'1d',['close','high'])
    # stock_price = attribute_history(stock, 1, '1m', 'close')
    # highest = stock_data.close.max()
    # profit = highest*(1-g.take_profit) # 移动止盈线
    # MACD
    dif, dea, macd = MACD(stock, check_date=context.current_dt, SHORT=12, LONG=29, MID=7, unit='1d')
    # KDJ
    K, D, J = KDJ(stock, check_date=context.current_dt, unit='1d', N=9, M1=3, M2=3)
    # 连续num_days分数变化
    # stock_name = get_security_info(stock).display_name
    # print(f'今日自选股是{stock_name}')
    stock_dif = g.stock_df[stock].diff().dropna()
    sig = np.sum((stock_dif < 0).astype(int))  # 如果连续num_days日下降即sig=num_days，卖出
    print(f'连续下降{sig}日')
    # 综合判断所有信号:大盘信号 + 个股信号
    if sig < 2:  # rsrs_score > g.score_threshold and today_MA > before_MA and sig < 2 :
        print('BUY')
        return "BUY"
    elif sig >= 2:  # (rsrs_score < -g.score_threshold and today_MA < before_MA) or sig >= 2:
        print('SELL')
        return "SELL"
    else:
        print('KEEP')
        return "KEEP"


# 3-1 过滤模块-过滤停牌股票
# 输入选股列表，返回剔除停牌股票后的列表
def filter_paused_stock(stock_list):
    current_data = get_current_data()
    return [stock for stock in stock_list if not current_data[stock].paused]


# 3-2 过滤模块-过滤ST及其他具有退市标签的股票
# 输入选股列表，返回剔除ST及其他具有退市标签股票后的列表
def filter_st_stock(stock_list):
    current_data = get_current_data()
    return [stock for stock in stock_list
            if not current_data[stock].is_st]


# 3-3 过滤模块-过滤涨停的股票
# 输入选股列表，返回剔除未持有且已涨停股票后的列表
def filter_limitup_stock(context, stock_list):
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    current_data = get_current_data()
    # 已存在于持仓的股票即使涨停也不过滤，避免此股票再次可买，但因被过滤而导致选择别的股票
    return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
            or last_prices[stock][-1] < current_data[stock].high_limit]


# 3-4 过滤模块-过滤跌停的股票
# 输入股票列表，返回剔除已跌停股票后的列表
def filter_limitdown_stock(context, stock_list):
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    current_data = get_current_data()
    return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
            or last_prices[stock][-1] > current_data[stock].low_limit]


# 4-1 交易模块-自定义下单
# 报单成功返回报单(不代表一定会成交),否则返回None,应用于
def order_target_value_(security, value):
    if value == 0:
        log.debug("Selling out %s" % (security))
    else:
        log.debug("Order %s to value %f" % (security, value))
    # 如果股票停牌，创建报单会失败，order_target_value 返回None
    # 如果股票涨跌停，创建报单会成功，order_target_value 返回Order，但是报单会取消
    # 部成部撤的报单，聚宽状态是已撤，此时成交量>0，可通过成交量判断是否有成交
    return order_target_value(security, value)


# 4-2 交易模块-开仓
# 买入指定价值的证券,报单成功并成交(包括全部成交或部分成交,此时成交量大于0)返回True,报单失败或者报单成功但被取消(此时成交量等于0),返回False
def open_position(security, value):
    order = order_target_value_(security, value)
    if order != None and order.filled > 0:
        return True
    return False


# 4-3 交易模块-平仓
# 卖出指定持仓,报单成功并全部成交返回True，报单失败或者报单成功但被取消(此时成交量等于0),或者报单非全部成交,返回False
def close_position(position):
    security = position.security
    if position.total_amount != 0:
        order = order_target_value_(security, 0)  # 可能会因停牌失败
    else:
        print(f'目前没有持有{get_security_info(security).display_name}')
        return position
    if order != None:
        if order.status == OrderStatus.held and order.filled == order.amount:
            return True
    return False


# 4-4 交易模块-调仓
# 当择时信号为买入时开始调仓，输入过滤模块处理后的股票列表，执行交易模块中的开平仓操作
def adjust_position(context, buy_stock):
    for stock in context.portfolio.positions:
        if stock != buy_stock:
            log.info("[%s]已不在应买入列表中" % (stock))
            position = context.portfolio.positions[stock]
            close_position(position)
        else:
            log.info("[%s]已经持有无需重复买入" % (stock))
    # 根据股票数量分仓
    # 此处只根据可用金额平均分配购买，不能保证每个仓位平均分配
    position_count = len(context.portfolio.positions)
    if g.stock_tobuy > position_count:
        value = context.portfolio.cash / (g.stock_tobuy - position_count)
        if context.portfolio.positions[buy_stock].total_amount == 0:
            open_position(buy_stock, value)


# 4-5 交易模块-择时交易
# 结合择时模块综合信号进行交易
def my_trade(context):
    # if context.current_dt.minute != 35:
    #    return
    # 以下的代码每小时跑一次
    # 获取选股列表并过滤掉:st,st*,退市,涨停,跌停,停牌
    stock_pool = get_stock_pool()
    check_out_list, _ = get_rank(stock_pool, g.run_today_day, context)
    print(f'check_out_list is {check_out_list}')
    g.stock_df = rank_stock_change(check_out_list, context)
    # check_out_list = filter_st_stock(check_out_list)
    # check_out_list = filter_limitup_stock(context, check_out_list)
    check_out_list = filter_limitdown_stock(context, check_out_list)
    check_out_list = filter_paused_stock(check_out_list)

    if not check_out_list:  # empoty list is False
        print(f'Stock is limit up or limit down.')
    else:
        # print('今日自选股:{}'.format(get_security_info(check_out_list[0]).display_name))
        print(f'今日自选股:{get_security_info(check_out_list[0]).display_name}')
        # 获取综合择时信号
        for stock in check_out_list:
            timing_signal = get_timing_signal(stock, context)
            name = get_security_info(stock).display_name
            print(f'{name} 今日择时信号:{timing_signal}')
            # 开始交易
            if timing_signal == 'SELL':
                for stock_pos in context.portfolio.positions:
                    if stock_pos not in check_out_list:
                        log.info("旧龙头已不再买入列表，卖出")
                        position = context.portfolio.positions[stock_pos]
                        close_position(position)
                    else:
                        position = context.portfolio.positions[stock]
                        close_position(position)
            elif timing_signal == 'BUY' or timing_signal == 'KEEP':
                adjust_position(context, stock)
                break  # only buy one stock
            else:
                pass


# 4-6 交易模块-止损
# 检查持仓并进行必要的止损操作
def check_lose(context):
    for position in list(context.portfolio.positions.values()):
        securities = position.security
        cost = position.avg_cost
        price = position.price
        ret = 100 * (price / cost - 1)
        value = position.value
        amount = position.total_amount
        # 这里设定15%止损
        if ret <= -15:
            order_target_value(position.security, 0)
            print("！！！！！！触发止损信号: 标的={},标的价值={},浮动盈亏={}% ！！！！！！"
                  .format(securities, format(value, '.2f'), format(ret, '.2f')))
            log.info('亏死了，溜溜溜')


# 4-7 交易模块-止盈
# 根据移动止盈线止盈
def check_profit(context):
    for stock in context.portfolio.positions:
        position = context.portfolio.positions[stock]
        security = position.security
        price = attribute_history(security, 1, '1m', 'close')
        highest = attribute_history(security, g.sec_data_num, '1d', 'high')
        if price.close[-1] < highest.high.max() * (1 - g.take_profit):
            print('触发止盈，卖卖卖')
            close_position(position)


# 5-1 复盘模块-打印
# 打印每日持仓信息
def print_trade_info(context):
    # 打印当天成交记录
    trades = get_trades()
    for _trade in trades.values():
        print('成交记录：' + str(_trade))
    # 打印账户信息
    for position in list(context.portfolio.positions.values()):
        securities = position.security
        cost = position.avg_cost
        price = position.price
        ret = 100 * (price / cost - 1)
        value = position.value
        amount = position.total_amount
        print('代码:{}'.format(securities))
        print('成本价:{}'.format(format(cost, '.2f')))
        print('现价:{}'.format(price))
        print('收益率:{}%'.format(format(ret, '.2f')))
        print('持仓(股):{}'.format(amount))
        print('市值:{}'.format(format(value, '.2f')))
    print('一天结束')
    print('———————————————————————————————————————分割线————————————————————————————————————————')