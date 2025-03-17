# encoding:gbk
'''
声明：源代码只用作学习使用，不做商业用途
小果聚宽跟单交易系统
作者：小果
时间：20240120
教程 https://gitee.com/li-xingguo11111/joinquant_trader_miniqmt
大qmt https://gitee.com/li-xingguo11111/joinquant_trader_bigqmt
实盘时间设置在下面的text里面改
"交易时间段":4,
"交易开始时间":9,
"交易结束时间":14,
测试时间设置
"时间设置":"时间设置********",
"交易时间段":8,
"交易开始时间":0,
"交易结束时间":24,
"是否参加集合竞价":"否",
"开始交易分钟":0,
实盘需要把txet下面的参数是否测试改成否
"是否开启测试":"否",
下面的参数测试的时候改成否,实盘改成是
"是否开启临时id记录":"是"
高频使用循环模式，低频使用定时
'''
import pandas as pd
import numpy as np
import talib
import requests
import json
from datetime import datetime
import math
import time
import random

text = {
    "账户": "99085312",
    "账户类型": "STOCK",
    "聚宽跟单": "跟单原理",
    "服务器设置": "服务器跟单设置",
    "服务器": "http://115.175.23.7",
    "端口": "2000",
    "测试说明": "开启测试就是选择历史交易不开启就是选择今天的数据",
    "是否开启测试": "否",
    "测试数量": 100,
    "跟单设置": "跟单设置***********",
    "账户跟单比例": 1,
    "多策略用逗号隔开": "多策略用逗号隔开********",
    "组合名称": ["低回撤搅屎棍组合策略", "高收益小市值组合策略", "bska1"],
    "组合授权码": ["低回撤搅屎棍组合策略", "高收益小市值组合策略", "bska1"],
    "组合跟单比例": [1, 1, 0.1],
    "不同策略间隔更新时间": 0,
    "下单默认说明": "默认/金额/数量",
    "下单模式": "默认",
    "下单值": 1000,
    "时间设置": "时间设置********",
    "交易时间段": 4,
    "交易开始时间": 9,
    "交易结束时间": 15,
    "是否参加集合竞价": "否",
    "开始交易分钟": 0,
    "是否开启临时id记录": "是",
    "发送微信消息": "是",
    "发送钉钉消息": "是"
}


# 记录临时id,避免循环下没有用的单子
class a:
    pass


a = a()
a.log_id = []


def init(c):
    # 账户
    c.account = text['账户']
    # 账户类型
    c.account_type = text['账户类型']
    c.url = text['服务器']
    c.port = text['端口']
    print('小果服务器提供数据支持************服务器{} 端口{}'.format(c.url, c.port))
    # 定时模式
    # c.run_time("update_all_data","1nDay","2024-07-25 09:45:00")
    # c.run_time("update_all_data","1nDay","2024-07-25 14:45:00")
    # 循环模式3秒
    c.run_time("update_all_data", "1nSecond", "2024-07-25 13:20:00")
    # c.run_time("tarder_test","3nSecond","2024-07-25 13:20:00")
    print(get_account(c, c.account, c.account_type))
    print(get_position(c, c.account, c.account_type))


# print(update_all_data(c))
def handlebar(c):
    pass


def tarder_test(c):
    print('交易测试***************')
    stock = '513100.SH'
    amount = 100
    maker = '交易测试'
    passorder(23, 1101, c.account, stock, 5, 0, amount, maker, 1, maker, c)


def get_del_buy_sell_data(c, name='测试1', password='123456'):
    '''
    处理交易数据获取原始数据
    '''
    test = text['是否开启测试']
    url = text['服务器']
    port = text['端口']
    now_date = str(datetime.now())[:10]
    xg_data = xg_jq_data(url=url, port=port, password=password)
    info = xg_data.get_user_data(data_type='用户信息')
    df = xg_data.get_user_data(data_type='实时数据')
    # print('用户信息已经读取')
    # print(info)
    if df.shape[0] > 0:
        stats = df['数据状态'].tolist()[-1]
        if stats == True:
            df['证券代码'] = df['股票代码'].apply(
                lambda x: str(x).split('.')[0] + '.SH' if str(x).split('.')[-1] == 'XSHG' else str(x).split('.')[
                                                                                                   0] + '.SZ')
            df['数据长度'] = df['证券代码'].apply(lambda x: len(str(x)))
            df['订单添加时间'] = df['订单添加时间'].apply(lambda x: str(x)[:10])
            df = df[df['数据长度'] >= 6]
            df['交易类型'] = df['买卖'].apply(lambda x: 'buy' if x == True else 'sell')
            df = df.drop_duplicates(subset=['股票代码', '下单数量', '买卖', '多空'], keep='last')
            df['组合名称'] = str(name)
            df['组合授权码'] = str(password)
            df['策略名称'] = str('聚宽跟单')
            df['订单ID'] = df['订单ID'].astype(str)
            df['证券代码'] = df['证券代码'].apply(lambda x: str(x))
            df['投资备注'] = df['组合授权码'] + ',' + df['订单ID']
            if test == '是':
                print('开启测试模式,实盘记得关闭')
                df = df
            else:
                df = df[df['订单添加时间'] == now_date]
        else:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    if df.shape[0] > 0:
        print('组合 {} 策略授权码 {} {}今天有跟单数据*********************'.format(name, password, now_date))
    # print(df)
    else:
        print('组合 {} 策略授权码 {} {}今天没有跟单数据*********************'.format(name, password, now_date))
    return df


# 发送微信消息
def send_wx_message(message, item='大同QMT实盘'):
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


def seed_dingding(message='买卖交易成功',
                  access_token_list=['2615283ba0ab84900235e4ecc262671a0c79f3fb13acdf35f22cfd4b0407295f']):
    access_token = random.choice(access_token_list)
    url = 'https://oapi.dingtalk.com/robot/send?access_token={}'.format(access_token)
    headers = {'Content-Type': 'application/json;charset=utf-8'}
    data = {
        "msgtype": "text",  # 发送消息类型为文本
        "at": {
            # "atMobiles": reminders,
            "isAtAll": False,  # 不@所有人
        },
        "text": {
            "content": '大同QMT交易通知\n' + message,  # 消息正文
        }
    }
    r = requests.post(url, data=json.dumps(data), headers=headers)
    text = r.json()
    errmsg = text['errmsg']
    if errmsg == 'ok':
        print('钉钉发送成功')
        return text
    else:
        print(text)
        return text


def get_trader_data(c, name='测试', password='123456', zh_ratio=0.1):
    '''
    获取交易数据
    组合的跟单比例
    '''
    test = text['是否开启测试']
    adjust_ratio = text['账户跟单比例']
    is_open_id_log = text['是否开启临时id记录']
    # 读取跟单数据
    df = get_del_buy_sell_data(c, name=name, password=password)
    try:
        df['证券代码'] = df['证券代码'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
    except:
        pass
    trader_log = get_order(c, c.account, c.account_type)
    now_date = str(datetime.now())[:10]
    # 剔除撤单废单
    not_list = [54, 57]
    if trader_log.shape[0] > 0:
        trader_log['撤单'] = trader_log['委托状态'].apply(lambda x: '是' if x in not_list else '不是')
        trader_log = trader_log[trader_log['撤单'] == '不是']
    else:
        trader_log = trader_log
    if trader_log.shape[0] > 0:
        trader_log['证券代码'] = trader_log['证券代码'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
        trader_log['组合授权码'] = trader_log['投资备注'].apply(lambda x: str(x).split(',')[0])
        trader_log['订单ID'] = trader_log['投资备注'].apply(lambda x: str(x).split(',')[-1])
        trader_log['订单ID'] = trader_log['订单ID'].astype(str)
        if test == '是':
            trader_log = trader_log
        else:
            trader_log = trader_log
        trader_log['组合授权码'] = trader_log['组合授权码'].astype(str)
        trader_log_1 = trader_log[trader_log['组合授权码'] == password]
        if trader_log_1.shape[0] > 0:
            trader_id_list = trader_log_1['订单ID'].tolist()
        else:
            trader_id_list = []
    else:
        trader_id_list = []
    if is_open_id_log == '是':
        for trader_id in a.log_id:
            trader_id_list.append(trader_id)
    else:
        pass
    trader_id_list = list(set(trader_id_list))

    if df.shape[0] > 0:
        df['组合授权码'] = df['组合授权码'].astype(str)
        # df['订单ID ']=df['订单ID'].astype(str)
        df = df[df['组合授权码'] == password]
        if df.shape[0] > 0:
            df['账户跟单比例'] = adjust_ratio
            df['组合跟单比例'] = zh_ratio
            df['交易检查'] = df['订单ID'].apply(lambda x: '已经交易' if x in trader_id_list else '没有交易')
            df = df[df['交易检查'] == '没有交易']
            amount_list = []
            if df.shape[0] > 0:
                for stock, amount, trader_type in zip(df['证券代码'].tolist(), df['下单数量'].tolist(),
                                                      df['交易类型'].tolist()):
                    try:
                        price = get_price(c, stock=stock)
                        test = text['是否开启测试']
                        test_amount = text['测试数量']
                        down_type = text['下单模式']
                        down_value = text['下单值']
                        send_wx_msg = text['发送微信消息']
                        send_dd_msg = text['发送钉钉消息']
                        if test == '是':
                            value = test_amount * price
                        else:
                            if down_type == '默认':
                                value = price * amount * adjust_ratio * zh_ratio
                            elif down_type == '数量':
                                value = price * down_value * adjust_ratio * zh_ratio
                            elif down_type == '金额':
                                value = down_value
                            else:
                                value = price * amount * adjust_ratio * zh_ratio
                        if trader_type == 'buy':
                            try:
                                trader_type, amount, price = order_stock_value(c, c.account, c.account_type, stock,
                                                                               value, 'buy')
                                print(trader_type, amount, price)
                                if trader_type == 'buy' and amount >= 10:
                                    amount = adjust_amount(c, stock=stock, amount=amount)
                                    amount_list.append(amount)
                                else:
                                    amount_list.append(0)

                                msg = f'当前时刻--{str(datetime.now())[:19]}\n买入股票--{stock}\n股数--{amount}\n单价--{price}\n总价--{price * amount}'
                                if send_wx_msg == '是':
                                    send_wx_message(message=msg)
                                if send_dd_msg == '是':
                                    seed_dingding(message=msg)
                            except Exception as e:
                                print('组合{} 组合授权码{} {}买入有问题可能没有资金'.format(name, password, stock))
                                amount_list.append(0)
                        elif trader_type == 'sell':
                            try:
                                trader_type, amount, price = order_stock_value(c, c.account, c.account_type, stock,
                                                                               value, 'sell')
                                if trader_type == 'sell' and amount >= 10:
                                    amount = adjust_amount(c, stock=stock, amount=amount)
                                    amount_list.append(amount)
                                else:
                                    amount_list.append(0)

                                msg = f'当前时刻--{str(datetime.now())[:19]}\n卖出股票--{stock}\n股数--{amount}\n单价--{price}\n总价--{price * amount}'
                                if send_wx_msg == '是':
                                    send_wx_message(message=msg)
                                if send_dd_msg == '是':
                                    seed_dingding(message=msg)
                            except Exception as e:

                                print('组合{} 组合授权码{} {}卖出有问题可能没有持股'.format(name, password, stock))
                                amount_list.append(0)

                        else:
                            print('组合{} 组合授权码{} {}未知的交易类型'.format(name, password, stock))

                    except Exception as e:
                        print(e, stock, '有问题*************')
                        amount_list.append(0)

                df['数量'] = amount_list
                not_trader = df[df['数量'] <= 0]
                # 数量为0的不进入下单记录
                df = df[df['数量'] >= 10]
                # df=df[df['数量']>=0]
                print('下单股票池*************')
                print(df)
                print(
                    '下单数量为0的标的可能没有持股,可能账户没有资金等待下次成交########################################################')
                print(not_trader)
                trader_log = pd.concat([trader_log, df], ignore_index=True)
                trader_log = trader_log.drop_duplicates(subset=['订单添加时间', '订单ID', '组合授权码', '组合名称'],
                                                        keep='last')
            else:
                # print('{}组合没有需要下单股票******************'.format(name))
                df = pd.DataFrame()
        else:
            print('{}没有这个组合*************'.format(name))
            df = pd.DataFrame()

    else:
        # print('{}交易股票池没有数据*************'.format(name))
        df = pd.DataFrame()
    return df


def start_trader_on(c, name='测试1', password='123456', zh_ratio=0.1):
    '''
    开始下单
    '''
    is_open_id_log = text['是否开启临时id记录']
    df = get_trader_data(c, name, password=password, zh_ratio=zh_ratio)
    try:
        df['证券代码'] = df['证券代码'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
    except:
        pass
    if df.shape[0] > 0:
        df['证券代码'] = df['证券代码'].astype(str)
        # print(df['证券代码'])
        df['证券代码'] = df['证券代码'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
        # 先卖在买
        sell_df = df[df['交易类型'] == 'sell']
        if sell_df.shape[0] > 0:
            for stock, amount, maker, in zip(sell_df['证券代码'].tolist(), sell_df['数量'].tolist(),
                                             sell_df['投资备注'].tolist()):
                try:

                    price = get_price(c, stock)
                    passorder(24, 1101, c.account, str(stock), 5, 0, int(amount), str(maker), 1, str(maker), c)
                    print('组合{} 卖出标的{} 数量{} 价格{}'.format(name, stock, amount, price))
                    if is_open_id_log == '是':
                        trader_id = str(maker).split(',')[-1]
                        a.log_id.append(trader_id)
                    else:
                        pass

                except Exception as e:
                    print(e)
                    print('组合{} {}卖出有问题'.format(name, stock))
        else:
            print('{}组合没有符合调参的卖出数据'.format(name))
        buy_df = df[df['交易类型'] == 'buy']
        if buy_df.shape[0] > 0:
            for stock, amount, maker, in zip(buy_df['证券代码'].tolist(), buy_df['数量'].tolist(),
                                             buy_df['投资备注'].tolist()):
                try:

                    price = get_price(c, stock)
                    passorder(23, 1101, c.account, str(stock), 5, 0, int(amount), str(maker), 1, str(maker), c)
                    # passorder(23, 1101, c.account, stock, 5, 0, int(amount), '',1,'',c)
                    print('组合{} 买入标的{} 数量{} 价格{}'.format(name, stock, amount, price))
                    if is_open_id_log == '是':
                        trader_id = str(maker).split(',')[-1]
                        a.log_id.append(trader_id)
                    else:
                        pass

                except Exception as e:
                    print(e)
                    print('组合{} {}买入有问题'.format(name, stock))

        else:
            print('{}组合没有符合调参的买入数据'.format(name))
    # else:
    #     print('{}组合没有符合调参数据'.format(name))


# print(a.log_id)
def update_all_data(c):
    '''
    更新策略数据
    '''
    if check_is_trader_date_1():
        name_list = text['组合名称']
        password_list = text['组合授权码']
        ratio_list = text['组合跟单比例']
        update_time = text['不同策略间隔更新时间']
        for name, password, ratio in zip(name_list, password_list, ratio_list):
            print('【【【【【【【策略---{}---分隔符】】】】】】】】'.format(name))
            start_trader_on(c, name=name, password=password, zh_ratio=ratio)
            time.sleep(update_time * 60)
    else:
        print('跟单{} 目前不是交易时间***************'.format(datetime.now()))
        time.sleep(30)


def order_stock_value(c, accountid, datatype, stock, value, trader_type):
    '''
    价值下单函数
    '''
    price = get_price(c, stock)
    hold_stock = get_position(c, accountid, datatype)
    if hold_stock.shape[0] > 0:
        hold_stock = hold_stock[hold_stock['持仓量'] >= 10]
        if hold_stock.shape[0] > 0:
            hold_df = hold_stock[hold_stock['证券代码'] == stock]
            if hold_df.shape[0] > 0:
                hold_amount = hold_df['持仓量'].tolist()[-1]
                av_amount = hold_df['可用数量'].tolist()[-1]
            else:
                hold_amount = 0
                av_amount = 0
        else:
            hold_amount = 0
            av_amount = 0
    else:
        hold_amount = 0
        av_amount = 0
    account = get_account(c, accountid, datatype)
    av_cash = account['可用金额']
    amount = value / price
    if str(stock)[:2] in ['11', '12']:
        amount = int(amount / 10) * 10
    else:
        amount = int(amount / 100) * 100
    if trader_type == 'buy':
        if av_cash >= value and amount >= 10:
            print('金额下单可以资金{}大于买入金额{} 买入{} 价格{} 数量{}'.format(av_cash, value, stock, price, amount))
            return 'buy', amount, price
        else:
            print(
                '金额下单可以资金{}小于买入金额{} 不买入{} 价格{} 数量{}'.format(av_cash, value, stock, price, amount))
            return '', '', price
    elif trader_type == 'sell':
        if av_amount >= amount and amount >= 10:
            print('金额下单 持有数量{} 可用数量{} 大于卖出数量{} 卖出{} 价格{} 数量{}'.format(hold_amount, av_amount,
                                                                                              amount, stock, price,
                                                                                              amount))
            return 'sell', amount, price
        elif av_amount < amount and av_amount >= 10:
            print(
                '金额下单 持有数量{} 可用数量{} 小于卖出数量{}，可以数量大于10 卖出{} 价格{} 数量{}'.format(hold_amount,
                                                                                                           av_amount,
                                                                                                           amount,
                                                                                                           stock, price,
                                                                                                           amount))
            return 'sell', amount, price
        else:
            print('金额下单 持有数量{} 可用数量{} 小于卖出数量{}，不卖出{} 价格{} 数量{}'.format(hold_amount, av_amount,
                                                                                                amount, stock, price,
                                                                                                amount))
            return 'sell', amount, price
    else:
        print('金额下单未知的交易类型{}'.format(stock))
        return '', amount, price


def buy(c, stock, price, amount, name):
    '''
    买入函数
    '''
    passorder(23, 1101, c.account, str(stock), 5, 0, amount, name, 1, name, c)


def sell(c, stock, price, amount, name):
    '''
    自定义卖出函数
    '''
    passorder(24, 1101, c.account, str(stock), 5, 0, amount, name, 1, name, c)


def get_price(c, stock):
    '''
    获取最新价格
    '''
    tick = c.get_full_tick(stock_code=[stock])
    tick = tick[stock]
    price = tick['lastPrice']
    return price


def adjust_amount(c, stock='', amount=''):
    '''
    调整数量
    '''
    if stock[:3] in ['110', '113', '123', '127', '128', '111'] or stock[:2] in ['11', '12']:
        amount = math.floor(amount / 10) * 10
    else:
        amount = int(round(amount, -2))
    return amount


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


class xg_jq_data:
    def __init__(self, url='http://124.220.32.224', port=8025, password='123456'):
        '''
        获取服务器数据
        '''
        self.url = url
        self.port = port
        self.password = password

    def get_user_data(self, data_type='用户信息'):
        '''
        获取使用的数据
        data_type='用户信息','实时数据',历史数据','清空实时数据','清空历史数据'
        '''
        url = '{}:{}/_dash-update-component'.format(self.url, self.port)
        headers = {'Content-Type': 'application/json'}
        data = {
            "output": "joinquant_trader_table.data@63d85b6189e42cba63feea36381da615c31ad8e36ae420ed67f60f3598efc9ad",
            "outputs": {"id": "joinquant_trader_table",
                        "property": "data@63d85b6189e42cba63feea36381da615c31ad8e36ae420ed67f60f3598efc9ad"},
            "inputs": [{"id": "joinquant_trader_password", "property": "value", "value": self.password},
                       {"id": "joinquant_trader_data_type", "property": "value", "value": data_type},
                       {"id": "joinquant_trader_text", "property": "value",
                        "value": "\n               {'状态': 'held', '订单添加时间': 'datetime.datetime(2024, 4, 23, 9, 30)', '买卖': 'False', '下单数量': '9400', '已经成交': '9400', '股票代码': '001.XSHE', '订单ID': '1732208241', '平均成交价格': '10.5', '持仓成本': '10.59', '多空': 'long', '交易费用': '128.31'}\n                "},
                       {"id": "joinquant_trader_run", "property": "value", "value": "运行"},
                       {"id": "joinquant_trader_down_data", "property": "value", "value": "不下载数据"}],
            "changedPropIds": ["joinquant_trader_run.value"], "parsedChangedPropsIds": ["joinquant_trader_run.value"]}
        res = requests.post(url=url, data=json.dumps(data), headers=headers)
        text = res.json()
        df = pd.DataFrame(text['response']['joinquant_trader_table']['data'])
        return df
