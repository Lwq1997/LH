# encoding:gbk
'''
小果聚宽交易成交系统高速服务器2
作者：小果量化
微信:xg_quant
时间：20250507
优化内容
1:交易检查功能,自动补单
2:5分钟不成交撤单了重新下单
3：优化了交易算法，交易细节
教程 https://gitee.com/li-xingguo11111/joinquant_trader_miniqmt
大qmt https://gitee.com/li-xingguo11111/joinquant_trader_bigqmt
高频使用循环模式，低频使用定时
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
import requests
import json
from datetime import datetime
import math
import time
import random

text = {
    "账户": "99085312",
    "账户支持融资融券": "账户支持融资融券,账户类型STOCK/CREDIT",
    "账户类型": "STOCK",
    "聚宽跟单": "跟单原理",
    "服务器设置": "服务器跟单设置",
    "是否开启id记录": "是",
    # "服务器": "http://115.175.23.7",
    # "端口": "2000",
    # "测试说明": "开启测试就是选择历史交易不开启就是选择今天的数据",
    # "是否开启测试": "否",
    # "测试数量": 100,
    # "跟单设置": "跟单设置***********",
    # "账户跟单比例": 0.5,
    # "多策略用逗号隔开": "多策略用逗号隔开********",
    # "组合名称": ["8n", "一进二", "xjy_001"],
    # "组合授权码": ["8n", "789789", "xjy_001"],
    # "组合跟单比例": [1, 1, 1],
    "下单默认说明": "默认/金额/数量",
    "下单模式": "默认",
    "下单值": 1000,
    "时间设置": "时间设置********",
    "交易时间段": 8,
    "交易开始时间": 8,
    "交易结束时间": 24,
    "是否参加集合竞价": "是",
    "开始交易分钟": 0,
    "发送微信消息": "是",
    "发送钉钉消息": "是"
}
data = [
    {
        "服务器": "http://server.588gs.cn",
        "端口": "2000",
        "测试说明": "开启测试就是选择历史交易不开启就是选择今天的数据",
        "是否开启测试": "否",
        "测试数量": 100,
        "跟单设置": "跟单设置***********",
        "账户跟单比例": 1,
        "多策略用逗号隔开": "多策略用逗号隔开********",
        "组合名称": ["低回撤搅屎棍组合策略", "高收益小市值组合策略", "xjy_bska"],
        "组合授权码": ["低回撤搅屎棍组合策略", "高收益小市值组合策略", "xjy_bska"],
        "组合跟单比例": [1, 1, 1],
        "不同策略间隔更新时间": 0,
        "买入价格编码": 4,
        "卖出价格编码": 6,
        "黑名单说明": "黑名单里面的标的，不会买入也不会卖出",
        "黑名单": []
    },
    {
        "服务器": "http://106.54.211.231",
        "端口": "3333",
        "测试说明": "开启测试就是选择历史交易不开启就是选择今天的数据",
        "是否开启测试": "否",
        "测试数量": 100,
        "跟单设置": "跟单设置***********",
        "账户跟单比例": 1,
        "多策略用逗号隔开": "多策略用逗号隔开********",
        "组合名称": ["低回撤搅屎棍组合策略", "高收益小市值组合策略", "xjy_bska"],
        "组合授权码": ["低回撤搅屎棍组合策略", "高收益小市值组合策略", "xjy_bska"],
        "组合跟单比例": [1, 1, 1],
        "不同策略间隔更新时间": 0,
        "买入价格编码": 4,
        "卖出价格编码": 6,
        "黑名单说明": "黑名单里面的标的，不会买入也不会卖出",
        "黑名单": []
    },
    # 两个打板策略
    {
        "服务器": "http://server.588gs.cn",
        "端口": "2000",
        "测试说明": "开启测试就是选择历史交易不开启就是选择今天的数据",
        "是否开启测试": "否",
        "测试数量": 100,
        "跟单设置": "跟单设置***********",
        "账户跟单比例": 1,
        "多策略用逗号隔开": "多策略用逗号隔开********",
        "组合名称": ["打板ST策略", "1n2_bska", "8n_bska"],
        "组合授权码": ["打板ST策略", "1n2_bska", "8n_bska"],
        "组合跟单比例": [1, 0.5, 0.5],
        "不同策略间隔更新时间": 0,
        "买入价格编码": 3,
        "卖出价格编码": 7,
        "黑名单说明": "黑名单里面的标的，不会买入也不会卖出",
        "黑名单": []
    },
    {
        "服务器": "http://106.54.211.231",
        "端口": "3333",
        "测试说明": "开启测试就是选择历史交易不开启就是选择今天的数据",
        "是否开启测试": "否",
        "测试数量": 100,
        "跟单设置": "跟单设置***********",
        "账户跟单比例": 1,
        "多策略用逗号隔开": "多策略用逗号隔开********",
        "组合名称": ["打板ST策略", "1n2_bska", "8n_bska"],
        "组合授权码": ["打板ST策略", "1n2_bska", "8n_bska"],
        "组合跟单比例": [1, 0.5, 0.5],
        "不同策略间隔更新时间": 0,
        "买入价格编码": 3,
        "卖出价格编码": 7,
        "黑名单说明": "黑名单里面的标的，不会买入也不会卖出",
        "黑名单": []
    }
]


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
    if c.account_type == 'stock' or c.account_type == 'STOCK':
        c.buy_code = 23
        c.sell_code = 24
    else:
        # 融资融券
        c.buy_code = 33
        c.sell_code = 34
    for item in data:
        url = item['服务器']
        port = item['端口']
        print('\n 小果服务器提供数据支持************服务器【{}】 端口【{}】'.format(url, port))
    # 定时模式
    # c.run_time("update_all_data","1nDay","2024-07-25 09:45:00")
    # c.run_time("update_all_data","1nDay","2024-07-25 14:45:00")
    # 循环模式3秒
    c.run_time("update_all_data", "60nSecond", "2024-07-25 13:20:00")
    # c.run_time("tarder_test","3nSecond","2024-07-25 13:20:00")
    # 交易检查函数1分钟一次
    c.run_time("run_check_trader_func", "60nSecond", "2024-07-25 13:20:00")
    # 撤单了重新下单5分钟一次
    c.run_time("run_order_trader_func", "300nSecond", "2024-07-25 13:20:00")
    print(get_account(c, c.account, c.account_type))
    print(get_position(c, c.account, c.account_type))


def handlebar(c):
    pass


def tarder_test(c):
    print('交易测试***************')
    stock = '513100.SH'
    amount = 100
    maker = '交易测试'
    passorder(23, 1101, c.account, stock, 5, 0, amount, maker, 1, maker, c)


def get_del_buy_sell_data(c, name='测试1', password='123456', item=None):
    '''
    处理交易数据获取原始数据
    '''
    test = item['是否开启测试']
    url = item['服务器']
    port = item['端口']
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
            df['投资备注'] = df['组合授权码'] + ',' + df['证券代码'] + ',' + df['订单ID']
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
        print('\n 组合 【{}】 策略授权码 【{}】【{}】今天有跟单数据*********************'.format(name, password, now_date))
    # print(df)
    else:
        print('\n 组合 【{}】策略授权码【{}】【{}】今天没有跟单数据*********************'.format(name, password, now_date))
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


def get_trader_data(c, name='测试', password='123456', zh_ratio=0.1, item=None):
    '''
    获取交易数据
    组合的跟单比例
    '''
    test = item['是否开启测试']
    adjust_ratio = item['账户跟单比例']
    # 读取跟单数据
    df = get_del_buy_sell_data(c, name=name, password=password, item=item)
    try:
        df['证券代码'] = df['证券代码'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
    except:
        pass
    trader_log = get_order(c, c.account, c.account_type)
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
                        test = item['是否开启测试']
                        test_amount = item['测试数量']
                        down_type = text['下单模式']
                        down_value = text['下单值']
                        send_wx_msg = text['发送微信消息']
                        send_dd_msg = text['发送钉钉消息']
                        if test == '是':
                            value = price * test_amount * adjust_ratio * zh_ratio
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

                                msg = f'当前时刻--{str(datetime.now())[:19]}\n组合--{name}\n买入股票--{stock}\n股数--{amount}\n单价--{price}\n总价--{price * amount}'
                                if send_wx_msg == '是':
                                    send_wx_message(message=msg)
                                if send_dd_msg == '是':
                                    seed_dingding(message=msg)
                            except Exception as e:
                                print('组合【{}】 组合授权码【{}】【{}】买入有问题可能没有资金'.format(name, password, stock))
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

                                msg = f'当前时刻--{str(datetime.now())[:19]}\n组合--{name}\n卖出股票--{stock}\n股数--{amount}\n单价--{price}\n总价--{price * amount}'
                                if send_wx_msg == '是':
                                    send_wx_message(message=msg)
                                if send_dd_msg == '是':
                                    seed_dingding(message=msg)
                            except Exception as e:

                                print('组合【{}】 组合授权码【{}】 【{}】卖出有问题可能没有持股'.format(name, password, stock))
                                amount_list.append(0)

                        else:
                            print('组合【{}】 组合授权码【{}】 【{}】未知的交易类型'.format(name, password, stock))

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
                # print('【{}】组合没有需要下单股票******************'.format(name))
                df = pd.DataFrame()
        else:
            print('【{}】没有这个组合*************'.format(name))
            df = pd.DataFrame()

    else:
        # print('【{}】交易股票池没有数据*************'.format(name))
        df = pd.DataFrame()
    return df


def check_is_sell(c, accountid, datatype, stock='', amount=0):
    '''
    检查是否可以卖出
    '''
    position = get_position(c, accountid, datatype)
    if position.shape[0] > 0:
        position = position[position['证券代码'] == stock]
        if position.shape[0] > 0:
            position = position[position['持仓量'] >= 0]
            if position.shape[0] > 0:
                av_amount = position['可用数量'].tolist()[-1]
                if av_amount >= 10:
                    return True
                else:
                    print('【{}】 不能卖出可用数量【{}】 小于卖出数量【{}】'.format(stock, av_amount, amount))
                    return False
            else:
                print('【{}】 不能卖出，没有持股'.format(stock))
                return False
        else:
            print('【{}】 不能卖出，没有持股'.format(stock))
            return False
    else:
        print('【{}】 账户空仓'.format(stock))
        return False


def check_is_buy(c, accountid, datatype, stock='', amount=0, price=0):
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
        print('【{}】 账户可用资金【{}】 不能买入价格【{}】 数量【{}】 标的'.format(stock, av_cash, price, amount))
        return False


def start_trader_on(c, name='测试1', password='123456', zh_ratio=0.1, item=None):
    '''
    开始下单
    '''
    is_open_id_log = text['是否开启id记录']
    del_trader_list = item['黑名单']
    sell_price_code = item['卖出价格编码']
    buy_price_code = item['买入价格编码']

    hold_stock = get_position(c, c.account, c.account_type)
    if hold_stock.shape[0] > 0:
        hold_stock = hold_stock[hold_stock['持仓量'] >= 10]
        if hold_stock.shape[0] > 0:
            hold_stock_list = hold_stock['证券代码'].tolist()
        else:
            hold_stock_list = []
    else:
        hold_stock_list = []
    df = get_trader_data(c, name, password=password, zh_ratio=zh_ratio, item=item)
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
        if len(hold_stock_list) > 0:
            av_amount_dict = dict(zip(hold_stock['证券代码'].tolist(), hold_stock['可用数量'].tolist()))
        else:
            av_amount_dict = {}
        if sell_df.shape[0] > 0:
            sell_df['可用数量'] = sell_df['证券代码'].apply(lambda x: av_amount_dict.get(x, 0))
            for stock, av_amount, amount, maker, in zip(sell_df['证券代码'].tolist(),
                                                        sell_df['可用数量'].tolist(),
                                                        sell_df['数量'].tolist(),
                                                        sell_df['投资备注'].tolist()):
                if stock not in del_trader_list:
                    if maker not in a.log_id:
                        # print('{} 标的不在黑名单卖出'.format(stock))
                        amount = amount if av_amount >= amount else av_amount
                        if amount >= 10:
                            try:
                                price = get_price(c, stock)
                                if check_is_sell(c, c.account, c.account_type, stock=stock, amount=amount):
                                    passorder(c.sell_code, 1101, c.account, str(stock), sell_price_code, 0,
                                              int(amount), str(maker), 1, str(maker), c)
                                    print('组合【{}】 卖出标的【{}】 数量【{}】 价格【{}】'.format(name, stock, amount, price))
                                    if is_open_id_log == '是':
                                        a.log_id.append(maker)
                                    else:
                                        pass
                                else:
                                    print('组合【{}】 【{}】不能卖出'.format(name, stock))
                            except Exception as e:
                                print(e)
                                print('组合【{}】 【{}】卖出有问题'.format(name, stock))
                        else:
                            print("【{}】 【{}】 卖出不了可用数量【{}】".format(name, stock, av_amount))
                    else:
                        print('【{}】 【{}】 在id记录不卖出，等待订单确认检查'.format(name, stock))
                else:
                    print('【{}】 【{}】 标的在黑名单不卖出'.format(name, stock))
        else:
            print('【{}】组合没有符合调参的卖出数据'.format(name))
        buy_df = df[df['交易类型'] == 'buy']
        if buy_df.shape[0] > 0:
            for stock, amount, maker, in zip(buy_df['证券代码'].tolist(), buy_df['数量'].tolist(),
                                             buy_df['投资备注'].tolist()):
                if stock not in del_trader_list:
                    # print('【{}】 标的不在黑名单买入'.format(stock))
                    if maker not in a.log_id:
                        try:
                            price = get_price(c, stock)
                            if check_is_buy(c, c.account, c.account_type, stock=stock, amount=amount, price=price):
                                passorder(c.buy_code, 1101, c.account, str(stock), buy_price_code, 0, int(amount),
                                          str(maker), 1, str(maker), c)
                                # passorder(23, 1101, c.account, stock, 5, 0, int(amount), '',1,'',c)
                                print('组合【{}】 买入标的【{}】 数量【{}】 价格【{}】'.format(name, stock, amount, price))
                                if c.is_open_id_log == '是':
                                    a.log_id.append(maker)
                                else:
                                    pass
                            else:
                                print('组合【{}】 【{}】不能买入'.format(name, stock))
                        except Exception as e:
                            print(e)
                            print('组合【{}】 【{}】买入有问题'.format(name, stock))
                    else:
                        print('【{}】 【{}】 在id记录不买入，等待订单确认检查'.format(name, stock))
                else:
                    print('【{}】 标的在黑名单不买入'.format(stock))

        else:
            print('【{}】组合没有符合调参的买入数据'.format(name))
    # else:
    #     print('【{}】组合没有符合调参数据'.format(name))


# print(a.log_id)
def update_all_data(c):
    '''
    更新策略数据
    '''
    if check_is_trader_date_1():
        for item in data:
            name_list = item['组合名称']
            password_list = item['组合授权码']
            ratio_list = item['组合跟单比例']
            update_time = item['不同策略间隔更新时间']
            for name, password, ratio in zip(name_list, password_list, ratio_list):
                print('【【【【【【【策略---【{}】-----IP---【{}】----端口---【{}】---分隔符】】】】】】】】'.format(name, item['服务器'],
                                                                                          item['端口']))
                start_trader_on(c, name=name, password=password, zh_ratio=ratio, item=item)
                time.sleep(update_time * 60)
    else:
        print('跟单【{}】 目前不是交易时间***************'.format(datetime.now()))
        time.sleep(30)


def run_check_trader_func(c):
    '''
    检查交易下单情况，没有下单的就补单
    '''
    trader_log = get_order(c, c.account, c.account_type)
    # 剔除撤单废单
    not_list = [54, 57]
    if trader_log.shape[0] > 0:
        trader_log['撤单'] = trader_log['委托状态'].apply(lambda x: '是' if x in not_list else '不是')
        trader_log = trader_log[trader_log['撤单'] == '不是']

    if trader_log.shape[0] > 0:
        trader_log['证券代码'] = trader_log['证券代码'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
        trader_log['组合授权码'] = trader_log['投资备注'].apply(lambda x: str(x).split(',')[0])
        trader_log['订单ID'] = trader_log['投资备注'].apply(lambda x: str(x).split(',')[-1])
        trader_log['订单ID'] = trader_log['订单ID'].astype(str)
        maker_list = trader_log['投资备注'].tolist()
    else:
        maker_list = []
    # 交易id记录a
    if len(a.log_id) > 0:
        for maker in a.log_id:
            if maker not in maker_list:
                a.log_id.remove(maker)
                print('【{}】 id记录没有委托重新委托*******************************'.format(maker))
            else:
                print('【{}】 id记录已经委托不委托*******************************'.format(maker))
    else:
        print('交易检查没有id记录数据*******************************')


def run_order_trader_func(c, item=None):
    '''
    下单不成交撤单在下单
    '''
    trader_log = get_order(c, c.account, c.account_type)
    # 不成交代码
    not_list = [49, 50, 51, 52]
    if trader_log.shape[0] > 0:
        trader_log['不成交'] = trader_log['委托状态'].apply(lambda x: '是' if x in not_list else '不是')
        trader_log = trader_log[trader_log['不成交'] == '是']
    for item in data:
        name_list = item['组合名称']
        password_list = item['组合授权码']
        sell_price_code = item['卖出价格编码']
        buy_price_code = item['买入价格编码']
        if trader_log.shape[0] > 0:
            trader_log['证券代码'] = trader_log['证券代码'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
            trader_log['组合授权码'] = trader_log['投资备注'].apply(lambda x: str(x).split(',')[0])
            trader_log['订单ID'] = trader_log['投资备注'].apply(lambda x: str(x).split(',')[-1])
            trader_log['订单ID'] = trader_log['订单ID'].astype(str)
            for name, password in zip(name_list, password_list):
                trader_log_new = trader_log[trader_log['组合授权码'] == password]
                if trader_log_new.shape[0] > 0:
                    for stock, amount, trader_type, maker, oder_id in zip(trader_log_new['证券代码'].tolist(),
                                                                          trader_log_new['未成交数量'].tolist(),
                                                                          trader_log_new['买卖方向'].tolist(),
                                                                          trader_log_new['投资备注'].tolist(),
                                                                          trader_log_new['订单编号'].tolist()):
                        price = get_price(c, stock)
                        # 未成交卖出
                        print(
                            '证券代码：【{}】 未成交数量【{}】交易类型【{}】 投资备注【{}】 订单id【{}】'.format(stock, amount, trader_type,
                                                                                            maker, oder_id))
                        if trader_type == 49:
                            # 撤单重新卖
                            cancel(oder_id, c.account, c.account_type, c)
                            passorder(c.sell_code, 1101, c.account, str(stock), sell_price_code, 0, int(amount),
                                      str(maker), 1, str(maker), c)
                            print('组合【{}】 撤单重新卖出标的【{}】 数量【{}】 价格【{}】'.format(name, stock, amount, price))
                        elif trader_type == 48:
                            # 撤单重新买
                            cancel(oder_id, c.account, c.account_type, c)
                            passorder(c.buy_code, 1101, c.account, str(stock), buy_price_code, 0, int(amount),
                                      str(maker),
                                      1, str(maker), c)
                            print('\n组合【{}】 撤单重新买入标的【{}】 数量【{}】 价格【{}】'.format(name, stock, amount, price))
                        else:
                            print('\n组合【{}】 撤单重新交易未知的交易类型'.format(name))
                else:
                    print('\n撤单了在下单组合【{}】没有委托数据'.format(name))
        else:
            print('\n撤单了重新下单没有委托数据')


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
    print('持股数量【{}】'.format(len(positions)))
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
    print('委托数量【{}】'.format(len(orders)))
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
    print('成交数量【{}】'.format(len(deals)))
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
