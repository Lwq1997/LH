# encoding:gbk
'''
西蒙斯高频分时网格策略实盘1
先模拟盘测试
作者:西蒙斯量化
微信:xg_quant
修改账户参数就可以改成自己的
"账户":"222",

数值	描述
-1	无效(只对于algo_passorder起作用)
0	卖5价
1	卖4价
2	卖3价
3	卖2价
4	卖1价
5	最新价
6	买1价
7	买2价(组合不支持)
8	买3价(组合不支持)
9	买4价(组合不支持)
10	买5价(组合不支持)
'''
import pandas as pd
import numpy as np
import talib
import time
from datetime import datetime

pd.set_option('display.float_format', lambda x: '%.2f' % x)

text = {
    "自定义交易品种交易": "自定义交易类型比如股票，可转债，etf***********",
    "账户支持融资融券": "账户支持融资融券,账户类型STOCK/CREDIT",
    "账户类型": "STOCK",
    "账户": "55001948",
    "账户类型": "STOCK",
    "是否测试说明": "实盘改成否",
    "是否测试": "否",
    "是否时间测试": "否",
    "测试时间": "20250425",
    "是否立马交易说明": "利用run_time运行策略第一次循环不会下单，但是会记录交易数据需要清空",
    "是否立马交易": "否",
    "买入价格编码": 5,
    "卖出价格编码": 5,
    "是否隔离策略": "否",
    "交易模式说明": "数量/金额",
    "交易模式": "数量",
    "固定交易数量": 100,
    "持有数量限制": 300,
    "固定交易金额": 200,
    "持有金额限制": 400,
    "网格模式说明": "百分比/ATR/",
    "网格模式": "ATR",
    "百分比模式设置": "百分比模式设置",
    "卖出单元格": 0.5,
    "买入单元格": -0.5,

    "ATR网格设置": "ATR网格设置********",
    "ATR周期": 14,
    "ATR数据周期": '1d',
    "是否自动生成ATR倍数": '否',
    "自定义固定ATR倍数": "自定义固定ATR倍数",
    "ATR倍数": 1,

    "时间设置": "时间设置********",
    "交易时间段": 8,
    "交易开始时间": 9,
    "交易结束时间": 24,
    "是否参加集合竞价": "是",
    "开始交易分钟": 0,
    "监测股票池设置": "监测股票池设置 自定义/持股",
    "监测股票池": "自定义",
    '自定义股票池': "自定义股票池设置",
    "股票池设置": "持有限制10的股票池设置",
    "自定义股票池": ['513100.SH', "159937.SZ", '511130.SH', '511090.SH',
                     '513400.SH', "512800.SH", '510300.SH', "515100.SH", '515450.SH',
                     "513500.SH"],
    "自定义股票池名称": ['纳斯达克ETF', '黄金ETF', '30年国债ETF', '30年国债ETF',
                         '道琼斯ETF', "银行ETF", '沪深30ETF', '红利ETF', '红利ETF',
                         "标普ETF"]

}


# 记录交易数据
class A:
    pass


a = A()


def init(c):
    # 账户
    c.account = text['账户']
    # 账户类型
    c.account_type = text['账户类型']
    c.atr_preiod = text['ATR数据周期']
    if c.account_type == 'stock' or c.account_type == 'STOCK':
        c.buy_code = 23
        c.sell_code = 24
    else:
        # 融资融券
        c.buy_code = 33
        c.sell_code = 34
    is_open = text['是否立马交易']
    if is_open == '是':
        a.del_log = True
    else:
        a.del_log = False
    c.buy_price_code = text['买入价格编码']
    c.sell_price_code = text['卖出价格编码']
    # 交易股票池
    a.trade_code_list = text['自定义股票池']
    a.trade_code_name = text['自定义股票池名称']
    c.stock_name_dict = dict(zip(a.trade_code_list, a.trade_code_name))
    a.log = get_order_log(c)
    print('开启策略交易）））））））））））））')
    print('读取系统的委托数据记录，继续上一步策略*******************************')
    print(a.log)
    print(get_account(c, c.account, c.account_type))
    print(get_position(c, c.account, c.account_type))
    # 3秒一次
    c.run_time("run_tarder_func", "3nSecond", "2024-07-25 13:20:00")
    # 1分钟下单了不成交撤单了在下
    c.run_time("run_order_trader_func", "60nSecond", "2024-07-25 13:20:00")


def handlebar(c):
    # run_tarder_func(c)
    pass


def get_order_log(c):
    '''
    第一次运行获取全部委托备注，避免程序断开没有记录数据
    '''
    order = get_order(c, c.account, c.account_type)
    if order.shape[0] > 0:
        result_list = []
        order['投资备注'] = order['投资备注'].apply(lambda x: str(x).split(','))
        for j in order['投资备注'].tolist():
            if len(j) == 7:
                result_list.append(j)
        if len(result_list) > 0:
            log = pd.DataFrame(result_list)
            log.columns = ['策略', '证券代码', '触发时间', '交易类型', '交易数量', '持有限制', '触发价格']
            log['触发时间'] = log['触发时间'].apply(lambda x: int(''.join(str(x)[10:][:9].split(':'))))
        else:
            log = pd.DataFrame()
    else:
        log = pd.DataFrame()
    return log


def get_now_tick_data(c, stock='511090.SH'):
    '''
    获取tick数据当天tick数据
    '''
    test = text['是否时间测试']
    test_date = text['测试时间']
    if test == '是':
        print('开启测试数据*************实盘记得关闭{}'.format(test_date))
        start_time = test_date
        end_time = start_time
    else:
        start_time = ''.join(str(datetime.now())[:10].split('-'))
        end_time = start_time
    hist = c.get_market_data_ex(
        fields=[],
        stock_code=[stock],
        period='tick',

        start_time=start_time,
        end_time=end_time,
        count=-1,
        fill_data=True,
        subscribe=True)
    hist = hist[stock]
    hist['date'] = hist.index.tolist()
    hist['date'] = hist['date'].astype(str)
    hist['date'] = hist['date'].apply(lambda x: int(str(x).split('.')[0][-6:]))
    # new_date=int(''.join(str(datetime.now())[10:][:9].split(':')))
    return hist


# return hist
def conditional_single_time_sharing_grid(c, stock='511090.SH', x1=0.2, x2=-0.2):
    '''
    条件单分时网格,基于实时连续tick数据计算
    stock_type=自定义/持股
    'time'                  #时间戳
    'lastPrice'             #最新价
    'open'                  #开盘价
    'high'                  #最高价
    'low'                   #最低价
    'lastClose'             #前收盘价
    'amount'                #成交总额
    'volume'                #成交总量
    'pvolume'               #原始成交总量
    'stockStatus'           #证券状态
    'openInt'               #持仓量
    'lastSettlementPrice'   #前结算
    'askPrice'              #委卖价
    'bidPrice'              #委买价
    'askVol'                #委卖量
    'bidVol'                #委买量
    'transactionNum'		#成交笔数
    '''
    name = '百分比高频分时网格策略实盘'
    now_date = datetime.now()
    # '证券代码','触发时间','触发的价格','资金类型','交易类型','交易数量','投资备注'
    trader_date = str(datetime.now())[:10]
    # 时间转数字
    new_date = int(''.join(str(datetime.now())[10:][:9].split(':')))
    # ['证券代码','触发时间','交易类型','交易数量','持有限制','触发价格']
    log = a.log
    tick = get_now_tick_data(c, stock=stock)
    base_price = tick['lastClose'].tolist()[-1]
    price = tick['lastPrice'].tolist()[-1]
    grid_type = text['网格模式']
    stock_name = c.stock_name_dict.get(stock, stock)
    if grid_type == 'ATR':
        name = 'ATR高频分时网格策略实盘'
        stock, price, atr_value, N, k, adjust_atr, atr_zdf = cacal_atr(c, stock)
        print(
            '时间:{} ,股票:{},价格:{},atr:{} ,周期:{} ,atr倍数:{} ,调整atr:{} ,atr对应涨跌幅:{}'.format(datetime.now(),
                                                                                                        stock, price,
                                                                                                        atr_value, N, k,
                                                                                                        adjust_atr,
                                                                                                        atr_zdf))
        x1 = atr_zdf
        x2 = -atr_zdf
    else:
        name = '百分比高频分时网格策略实盘'
    if log.shape[0] > 0:
        log['触发价格'] = pd.to_numeric(log['触发价格'])
        try:
            log['触发时间'] = log['触发时间'].apply(lambda x: int(''.join(str(x)[10:][:9].split(':'))))
        except:
            pass
        # print(log)
        log = log.sort_values(by='触发时间', ascending=True)
        log = log[log['证券代码'] == stock]
        if log.shape[0] > 0:
            # 上次交易触发的类型,上次触发卖出的类型
            shift_trader_type = log['交易类型'].tolist()[-1]
            # 触发时间
            cf_time = log['触发时间'].tolist()[-1]
            # 上次卖出，下次买入，卖出了继续上涨但是没有触发下次卖出条件
            if shift_trader_type == 'sell':
                # 触发后的数据
                tick = tick[tick['date'] >= cf_time]
                # 触发继续上涨的单元格n,卖出
                pre_price = log['触发价格'].tolist()[-1]
                n = ((price - pre_price) / pre_price) * 100
                ##触发继续上涨的单元格n,卖出
                if n >= x1:
                    pre_price = log['触发价格'].tolist()[-1]
                    print(
                        '{} {}触发二次连续卖出 目前网格涨跌幅{} 大于目前涨跌幅{}'.format(datetime.now(), stock, n, x1))
                else:
                    max_price = max(tick['lastPrice'].tolist())
                    max_price = max(max_price, pre_price)
                    print("{} {} 目前价格{} 上次触发价{} 目前网格涨跌幅{} 最高价{} 继续上涨的长度{}".format(
                        datetime.now(), stock, price, pre_price, n, max_price, max_price - pre_price))
                    pre_price = max_price
                    n = ((price - pre_price) / pre_price) * 100
            # 上次触发买入,下次触发卖出
            elif shift_trader_type == 'buy':
                # 触发后的数据
                tick = tick[tick['date'] >= cf_time]
                pre_price = log['触发价格'].tolist()[-1]
                n = ((price - pre_price) / pre_price) * 100
                # 触发了网格买入，继续触发下次买入网格
                if n <= x2:
                    pre_price = log['触发价格'].tolist()[-1]
                    print(
                        '{} {}触发二次连续买入 目前网格涨跌幅{} 大于目前涨跌幅{}'.format(datetime.now(), stock, n, x2))
                # 没有触发下次买入网格在期间波动
                else:
                    min_price = min(tick['lastPrice'].tolist())
                    min_price = min(min_price, pre_price)
                    print("{} {} 目前价格{} 上次触发价{} 目前网格涨跌幅{} 最低价{} 继续下跌的长度{}".format(
                        datetime.now(), stock, price, pre_price, n, min_price, min_price - pre_price))
                    pre_price = min_price
                    n = ((price - pre_price) / pre_price) * 100
            else:
                pre_price = base_price
                n = ((price - pre_price) / pre_price) * 100
            zdf = ((price - pre_price) / pre_price) * 100
        else:
            pre_price = base_price
            zdf = ((price - base_price) / base_price) * 100
            n = ((price - pre_price) / pre_price) * 100
    else:
        pre_price = base_price
        zdf = ((price - base_price) / base_price) * 100
        n = ((price - pre_price) / pre_price) * 100
    if zdf >= x1:
        print('{} 模块{} 卖出{}  目前涨跌幅{} 大于目前标涨跌幅{} '.format(now_date, name, stock, zdf, x1))
        return name, 'sell'
    elif zdf <= x2:
        print('{} 模块{} 买入{}  目前涨跌幅{} 小于目前标涨跌幅{} '.format(now_date, name, stock, zdf, x2))
        return name, 'buy'
    else:
        print('{} 模块{} 不符合交易{}  目前涨跌幅{} 目前标涨跌幅{}在区间波动 '.format(now_date, name, stock, zdf, x1))
        return name, ''


def ATR(CLOSE, HIGH, LOW, N=14):
    '''
    真实波幅
    输出MTR:(最高价-最低价)和1日前的收盘价-最高价的绝对值的较大值和1日前的收盘价-最低价的绝对值的较大值
    输出真实波幅:MTR的N日简单移动平均
    '''
    MTR = MAX(MAX((HIGH - LOW), ABS(REF(CLOSE, 1) - HIGH)), ABS(REF(CLOSE, 1) - LOW))
    ATR = MA(MTR, N)
    return MTR, ATR


def cacal_auto_k_volatility(atr, lookback=30):
    '''
    基于历史波动率自动计算K值
    '''
    # 计算近期ATR波动率
    recent_atr = atr[-lookback:]
    atr_std = recent_atr.std()
    atr_mean = recent_atr.mean()

    # 波动率聚类分析
    volatility_ratio = atr_std / atr_mean

    # 计算辅助指标
    skewness = pd.Series(atr).skew()
    kurtosis = pd.Series(atr).kurt()

    # 动态K值计算
    if volatility_ratio < 0.2:
        k = 0.8  # 低波动环境
    elif volatility_ratio < 0.5:
        k = 1.2  # 正常波动
    else:
        k = 1.8  # 高波动环境
    return k


def cacal_atr(c, stock='513100.SH'):
    '''
    计算atr
    '''
    N = text['ATR周期']
    is_open_auto_k = text['是否自动生成ATR倍数']
    hist = c.get_market_data_ex(
        fields=[],
        stock_code=[stock],
        period=c.atr_preiod,
        start_time='20240101',
        end_time='20500101',
        count=-1,
        fill_data=True,
        subscribe=True)
    hist = hist[stock]
    CLOSE = hist['close']
    HIGH = hist['high']
    LOW = hist['low']
    price = CLOSE.tolist()[-1]
    mtr, atr = ATR(CLOSE, HIGH, LOW, N)
    if is_open_auto_k == '是':
        k = cacal_auto_k_volatility(atr, lookback=30)
    else:
        k = text['ATR倍数']
    atr_value = atr.tolist()[-1]
    adjust_atr = atr_value * k
    atr_zdf = (adjust_atr / price) * 100
    return stock, price, atr_value, N, k, adjust_atr, atr_zdf


def check_is_sell(c, accountid, datatype, stock='513100.SH', amount=100):
    '''
    检查是否可以卖出
    '''
    position = get_position(c, accountid, datatype)
    if position.shape[0] > 0:
        position = position[position['证券代码'] == stock]
        if position.shape[0] > 0:
            position = position[position['持仓量'] >= 10]
            print(position)
            if position.shape[0] > 0:
                hold_amount = position['持仓量'].tolist()[-1]
                av_amount = position['可用数量'].tolist()[-1]
                if av_amount >= amount and amount >= 10:
                    return True
                elif av_amount < amount and av_amount >= 10:
                    return True
            else:
                return False
        else:
            return False
    else:
        return False


def check_is_buy(c, accountid, datatype, stock='513100.SH', amount=100, price=1.3):
    '''
    检查是否可以买入
    '''
    account = get_account(c, accountid, datatype)
    # 可以使用的现金
    av_cash = account['可用金额']
    value = amount * price
    if av_cash >= value:
        return True
    else:
        return False


def check_hold_limit(c, accountid, datatype, stock='513100.SH', limit=1000):
    '''
    检查是否到持股限制
    '''
    position = get_position(c, accountid, datatype)
    if position.shape[0] > 0:
        position = position[position['证券代码'] == stock]
        if position.shape[0] > 0:
            position = position[position['持仓量'] >= 10]
            if position.shape[0] > 0:
                hold_amount = position['持仓量'].tolist()[-1]
            else:
                hold_amount = 0
        else:
            hold_amount = 0
    else:
        hold_amount = 0
    av_amount = limit - hold_amount
    if av_amount >= 10:
        return True
    else:
        return False


def adjust_amount(c, stock='', amount=''):
    '''
    调整数量
    '''
    if stock[:3] in ['110', '113', '123', '127', '128', '111'] or stock[:2] in ['11', '12']:
        amount = math.floor(amount / 10) * 10
    else:
        amount = math.floor(amount / 100) * 100
    return amount


def run_tarder_func(c):
    '''
    运行交易函数
    '''
    down_type = text['交易模式']
    fix_amount = text['固定交易数量']
    hold_amount_limit = text['持有数量限制']
    fix_value = text['固定交易金额']
    hold_value_lilit = text['持有金额限制']
    stock_list_type = text['监测股票池']
    x1 = text['卖出单元格']
    x2 = text['买入单元格']
    test = text['是否测试']
    if check_is_trader_date_1():
        if test == '是':
            print('开启测试模式实盘记得关闭*（（（（（（（（（（（（（（（（（')
            a.log = pd.DataFrame()
        else:
            pass
        # print(a.log)
        now_date = datetime.now()
        if stock_list_type == '自定义':
            df = pd.DataFrame()
            df['证券代码'] = text['自定义股票池']
            df['证券名称'] = text['自定义股票池名称']
        else:
            df = get_position(c, c.account, c.account_type)
        if df.shape[0] > 0:
            for stock in df['证券代码'].tolist():
                try:
                    # if True:
                    price = get_price(c, stock)
                    if down_type == '数量':
                        fix_amount = fix_amount
                        hold_amount_limit = hold_amount_limit
                    else:
                        fix_amount = fix_value / price
                        fix_amount = adjust_amount(c, stock=stock, amount=fix_amount)
                        hold_amount_limit = hold_value_lilit / price
                        hold_amount_limit = adjust_amount(c, stock=stock, amount=hold_amount_limit)

                    name, trader_type = conditional_single_time_sharing_grid(c, stock=stock, x1=x1, x2=x2)
                    if trader_type == 'sell':

                        if check_is_sell(c, c.account, c.account_type, stock=stock, amount=fix_amount):
                            trader_type = 'sell'
                            amount = fix_amount
                            price = price
                        else:
                            print("{} {} 不能卖出".format(datetime.now(), stock))
                            trader_type = ''
                            amount = fix_amount
                            price = price
                    elif trader_type == 'buy':
                        # 检查是否到达持股限制
                        if check_hold_limit(c, c.account, c.account_type,
                                            stock=stock, limit=hold_amount_limit) == True:
                            # 检查是否可以买入
                            if check_is_buy(c, c.account, c.account_type, stock=stock, amount=fix_amount, price=price):

                                trader_type = 'buy'
                                amount = fix_amount
                                price = price
                            else:
                                trader_type = ''
                                amount = fix_amount
                                price = price
                                print("{} {} 买入不了".format(datetime.now(), stock))
                        else:
                            trader_type = ''
                            amount = fix_amount
                            price = price
                            print("{} {} 不买入超过持有限制".format(datetime.now(), stock))
                    else:
                        trader_type = ''
                        amount = fix_amount
                        price = price

                    if trader_type == 'buy' and amount >= 10:
                        # '证券代码','触发时间','交易类型','交易数量','持有限制,'触发价格''
                        flag = "{},{},{},{},{},{},{}".format(name, stock, now_date, 'buy', amount, hold_amount_limit,
                                                             price)
                        passorder(c.buy_code, 1101, c.account, str(stock), c.buy_price_code, 0, amount, flag, 1, flag,
                                  c)
                        print('{} {} {} 最新价格{} 买入{} 数量***************'.format(name, now_date, stock, price,
                                                                                      amount))
                    elif trader_type == 'sell' and amount >= 10:
                        flag = "{},{},{},{},{},{},{}".format(name, stock, now_date, 'sell', amount, hold_amount_limit,
                                                             price)
                        passorder(c.sell_code, 1101, c.account, str(stock), c.sell_price_code, 0, amount, flag, 1, flag,
                                  c)
                        print('{} {} {} 最新价格{} 卖出{} 数量*******************'.format(name, now_date, stock, price,
                                                                                          amount))
                    else:
                        print('{} {} {} 没有触发网格继续观察'.format(name, now_date, stock))
                    if (trader_type == 'buy' or trader_type == 'sell') and amount >= 10:
                        # '证券代码','触发时间','交易类型','交易数量','持有限制'
                        df1 = pd.DataFrame()
                        df1['策略'] = [name]
                        df1['证券代码'] = [stock]
                        df1['触发时间'] = [now_date]
                        df1['交易类型'] = [trader_type]
                        df1['交易数量'] = [amount]
                        df1['持有限制'] = [hold_amount_limit]
                        df1['触发价格'] = [price]
                        df1['触发时间'] = df1['触发时间'].apply(lambda x: int(''.join(str(x)[10:][:9].split(':'))))
                        a.log = pd.concat([a.log, df1], ignore_index=True)
                    else:
                        pass
                # print(a.log)

                except Exception as e:
                    print(e, stock, '{}运行有问题可能不是交易日期'.format(datetime.now()))

            if a.del_log == True:
                print('第一次循环清空交易记录********************************')
                a.log = pd.DataFrame()
                a.del_log = False
            else:
                a.del_log = False

        else:
            print('{} 分时网格股票没有数据'.format(now_date))
    else:
        print('{} 分时网格股票不是交易时间'.format(datetime.now()))


def run_order_trader_func(c):
    '''
    下单不成交撤单在下单
    '''
    trader_log = get_order(c, c.account, c.account_type)
    now_date = str(datetime.now())[:10]
    # 不成交代码,注意57这个是策略下的废单，看个人是否需要
    not_list = [49, 50, 51, 52, 57]
    if trader_log.shape[0] > 0:
        trader_log['不成交'] = trader_log['委托状态'].apply(lambda x: '是' if x in not_list else '不是')
        trader_log = trader_log[trader_log['不成交'] == '是']
    else:
        trader_log = trader_log
    name_list = ['ATR高频分时网格策略实盘', "百分比高频分时网格策略实盘"]
    try:
        trader_log = trader_log.drop_duplicates(subset=['投资备注'], keep='last')
    except Exception as e:
        trader_log = pd.DataFrame()
        print(e)
    print('******************委托')
    print(trader_log)
    if trader_log.shape[0] > 0:
        trader_log['证券代码'] = trader_log['证券代码'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
        trader_log['策略'] = trader_log['投资备注'].apply(lambda x: str(x).split(',')[0])
        trader_log['本策略'] = trader_log['策略'].apply(lambda x: '是' if x in name_list else '不是')
        trader_log = trader_log[trader_log['本策略'] == '是']
        if trader_log.shape[0] > 0:
            for stock, amount, trader_type, maker, oder_id, name in zip(trader_log['证券代码'].tolist(),
                                                                        trader_log['未成交数量'].tolist(),
                                                                        trader_log['买卖方向'].tolist(),
                                                                        trader_log['投资备注'].tolist(),
                                                                        trader_log['订单编号'].tolist(),
                                                                        trader_log['策略'].tolist()):
                price = get_price(c, stock)
                # 未成交卖出
                print('证券代码：{} 未成交数量{}交易类型{} 投资备注{} 订单id{}'.format(stock, amount, trader_type, maker,
                                                                                      oder_id))
                if trader_type == 49:
                    cancel(oder_id, c.account, c.account_type, c)
                    passorder(c.sell_code, 1101, c.account, str(stock), c.sell_price_code, 0, int(amount), str(maker),
                              1, str(maker), c)
                    print('组合{} 撤单重新卖出标的{} 数量{} 价格{}'.format(name, stock, amount, price))
                elif trader_type == 48:
                    cancel(oder_id, c.account, c.account_type, c)
                    passorder(c.buy_code, 1101, c.account, str(stock), c.buy_price_code, 0, int(amount), str(maker), 1,
                              str(maker), c)
                    print('组合{} 撤单重新买入标的{} 数量{} 价格{}'.format(name, stock, amount, price))
                else:
                    print('组合{} 撤单重新交易未知的交易类型'.format(name))
        else:
            print('撤单了在下单组合没有委托数据')
    else:
        print('撤单了重新下单没有委托数据')


def check_is_trader_date_1():
    '''
    检测是不是交易时间
    '''
    trader_time = text['交易时间段']
    start_date = text['交易开始时间']
    end_date = text['交易结束时间']
    start_mi = text['开始交易分钟']
    jhjj = text['是否参加集合竞价']
    if jhjj == '是':
        jhjj_time = 15
    else:
        jhjj_time = 30
    loc = time.localtime()
    tm_hour = loc.tm_hour
    tm_min = loc.tm_min
    wo = loc.tm_wday
    if wo <= trader_time:
        if tm_hour >= start_date and tm_hour <= end_date:
            if tm_hour == 9 and tm_min < jhjj_time:
                return False

            elif tm_min >= start_mi:
                return True
            else:
                return False
        else:
            return False
    else:
        print('周末')
        return False


def get_price(c, stock):
    '''
    获取最新价格
    '''
    tick = c.get_full_tick(stock_code=[stock])
    tick = tick[stock]
    price = tick['lastPrice']
    return price


# 获取账户总权益m_dBalance
def get_account(c, accountid, datatype):
    '''
    获取账户数据
    '''
    accounts = get_trade_detail_data(accountid, datatype, 'account')
    result = {}
    for dt in accounts:
        result['总资产'] = dt.m_dBalance
        result['净资产'] = dt.m_dAssureAsset
        result['总市值'] = dt.m_dInstrumentValue
        result['总负债'] = dt.m_dTotalDebit
        result['可用金额'] = dt.m_dAvailable
        result['盈亏'] = dt.m_dPositionProfit
    return result


# 获取持仓信息{code.market:手数}
def get_position(c, accountid, datatype):
    '''
    获取持股数据
    '''
    positions = get_trade_detail_data(accountid, datatype, 'position')
    data = pd.DataFrame()
    print('持股数量{}'.format(len(positions)))
    if len(positions) > 0:
        df = pd.DataFrame()
        try:
            for dt in positions:
                df['股票代码'] = [dt.m_strInstrumentID]
                df['市场类型'] = [dt.m_strExchangeID]
                df['证券代码'] = df['股票代码'] + '.' + df['市场类型']
                df['证券名称'] = [dt.m_strInstrumentName]
                df['持仓量'] = [dt.m_nVolume]
                df['可用数量'] = [dt.m_nCanUseVolume]
                df['成本价'] = [dt.m_dOpenPrice]
                df['市值'] = [dt.m_dInstrumentValue]
                df['持仓成本'] = [dt.m_dPositionCost]
                df['盈亏'] = [dt.m_dPositionProfit]
                data = pd.concat([data, df], ignore_index=True)
            df = data
            is_del = text['是否隔离策略']
            if is_del == '是':
                df['证券代码'] = df['证券代码'].astype(str)
                df['隔离策略'] = df['证券代码'].apply(lambda x: '是' if x in a.trade_code_list else '不是')
                df = df[df['隔离策略'] == '是']
                data = df
            else:
                data = data

        except Exception as e:
            print('获取持股隔离股票池有问题')
            data = pd.DataFrame()
    else:
        data = pd.DataFrame()
    return data


def get_order(c, accountid, datatype):
    '''
    获取委托
    '''
    data = pd.DataFrame()
    orders = get_trade_detail_data(accountid, datatype, 'order')
    print('委托数量{}'.format(len(orders)))
    if len(orders) > 0:
        df = pd.DataFrame()
        for o in orders:
            df['股票代码'] = [o.m_strInstrumentID]
            df['市场类型'] = [o.m_strExchangeID]
            df['证券代码'] = df['股票代码'] + '.' + df['市场类型']
            df['买卖方向'] = [o.m_nOffsetFlag]
            df['委托数量'] = [o.m_nVolumeTotalOriginal]
            df['成交均价'] = [o.m_dTradedPrice]
            df['成交数量'] = [o.m_nVolumeTraded]
            df['成交金额'] = [o.m_dTradeAmount]
            df['投资备注'] = [o.m_strRemark]
            df['委托状态'] = [o.m_nOrderStatus]
            df['委托数量'] = [o.m_nVolumeTotalOriginal]
            df['成交数量'] = [o.m_nVolumeTraded]
            df['订单编号'] = [o.m_strOrderSysID]
            df['未成交数量'] = df['委托数量'] - df['成交数量']
            data = pd.concat([data, df], ignore_index=True)
    else:
        data = pd.DataFrame()
    return data


def get_deal(c, accountid, datatype):
    '''
    获取成交
    '''
    data = pd.DataFrame()
    deals = get_trade_detail_data(account, 'stock', 'deal')
    print('成交数量{}'.format(len(deals)))
    if len(deals):
        df = pd.DataFrame()
        for dt in deals:
            df['股票代码'] = [dt.m_strInstrumentID]
            df['市场类型'] = [dt.m_strExchangeID]
            df['证券代码'] = df['股票代码'] + '.' + df['市场类型']
            df['证券名称'] = [dt.m_strInstrumentName]
            df['买卖方向'] = [dt.m_nOffsetFlag]
            df['成交价格'] = [dt.m_dPrice]
            df['成交数量'] = [dt.m_nVolume]
            df['成交金额'] = [dt.m_dTradeAmount]
            data = pd.concat([data, df], ignore_index=True)
    else:
        data = pd.DataFrame()


def order_target_amount(c, accountid, datatype, stock, price, target_amount, com_ratio=0.0001):
    '''
    目标交易数量
    '''
    account = get_account(c, accountid, datatype)
    # 可以使用的现金
    av_cash = account['可用金额']
    position = get_position(c, accountid, datatype)
    if position.shape[0] > 0:
        position[position['持仓量'] >= 10]
        if position.shape[0] > 0:
            hold_amount = position['持仓量'].tolist()[-1]
            av_amount = position['可用数量'].tolist()[-1]
        else:
            hold_amount = 0
            av_amount = 0
    else:
        hold_amount = 0
        av_amount = 0
    # 可以交易的数量
    av_trader_amount = target_amount - hold_amount
    # 存在买入空间
    if av_trader_amount >= 10:
        # 买入的价值
        value = av_trader_amount * price
        # 手续费
        com = value * com_ratio
        if av_cash >= value + com:
            print('{} 目标数量{} 持有数量{} 可用数量{} 买入数量{} 可用资金{} 大于买入资金{} 买入'.format(stock,
                                                                                                         target_amount,
                                                                                                         hold_amount,
                                                                                                         av_amount,
                                                                                                         av_trader_amount,
                                                                                                         av_cash,
                                                                                                         value))
            return 'buy', price, av_trader_amount
        else:
            print('{} 目标数量{} 持有数量{} 可用数量{} 买入数量{} 可用资金{} 小于买入资金{} 不买入'.format(stock,
                                                                                                           target_amount,
                                                                                                           hold_amount,
                                                                                                           av_amount,
                                                                                                           av_trader_amount,
                                                                                                           av_cash,
                                                                                                           value))
            return '', price, av_trader_amount
    elif av_trader_amount <= -10:
        av_trader_amount = abs(av_trader_amount)
        if av_amount >= av_trader_amount:
            print('{} 目标数量{} 持有数量{} 可用数量{}大于 卖出数量{} 卖出'.format(stock, target_amount, hold_amount,
                                                                                   av_amount, av_trader_amount))
            return 'sell', price, av_trader_amount
        else:
            print(
                '{} 目标数量{} 持有数量{} 可用数量{}小于 卖出数量{} 卖出全部'.format(stock, target_amount, hold_amount,
                                                                                     av_amount, av_trader_amount))
            return 'sell', price, av_amount
    else:
        print('{} 目标数量{} 持有数量{}一样不交易'.format(stock, target_amount, hold_amount))
        return '', '', ''


def RET(S, N=1):
    '''
    返回序列倒数第N个值,默认返回最后一个
    '''
    return np.array(S)[-N]


def ABS(S):
    '''
    返回N的绝对值
    '''
    return np.abs(S)


def MAX(S1, S2):
    '''
    序列max
    '''
    return np.maximum(S1, S2)


def MIN(S1, S2):
    '''
    序列min
    '''
    return np.minimum(S1, S2)


def IF(S, A, B):
    '''
    序列布尔判断 return=A  if S==True  else  B
    '''
    return np.where(S, A, B)


def REF(S, N=1):
    '''
    对序列整体下移动N,返回序列(shift后会产生NAN)
    '''
    return pd.Series(S).shift(N).values


def MA(S, N):
    '''
    求序列的N日简单移动平均值，返回序列
    '''
    return pd.Series(S).rolling(N).mean().values