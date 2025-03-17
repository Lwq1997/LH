# encoding:gbk
'''
������Դ����ֻ����ѧϰʹ�ã�������ҵ��;
С���ۿ��������ϵͳ
���ߣ�С��
ʱ�䣺20240120
�̳� https://gitee.com/li-xingguo11111/joinquant_trader_miniqmt
��qmt https://gitee.com/li-xingguo11111/joinquant_trader_bigqmt
ʵ��ʱ�������������text�����
"����ʱ���":4,
"���׿�ʼʱ��":9,
"���׽���ʱ��":14,
����ʱ������
"ʱ������":"ʱ������********",
"����ʱ���":8,
"���׿�ʼʱ��":0,
"���׽���ʱ��":24,
"�Ƿ�μӼ��Ͼ���":"��",
"��ʼ���׷���":0,
ʵ����Ҫ��txet����Ĳ����Ƿ���Ըĳɷ�
"�Ƿ�������":"��",
����Ĳ������Ե�ʱ��ĳɷ�,ʵ�̸ĳ���
"�Ƿ�����ʱid��¼":"��"
��Ƶʹ��ѭ��ģʽ����Ƶʹ�ö�ʱ
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
    "�˻�": "99085312",
    "�˻�����": "STOCK",
    "�ۿ����": "����ԭ��",
    "����������": "��������������",
    "������": "http://115.175.23.7",
    "�˿�": "2000",
    "����˵��": "�������Ծ���ѡ����ʷ���ײ���������ѡ����������",
    "�Ƿ�������": "��",
    "��������": 100,
    "��������": "��������***********",
    "�˻���������": 1,
    "������ö��Ÿ���": "������ö��Ÿ���********",
    "�������": ["�ͻس���ʺ����ϲ���", "������С��ֵ��ϲ���", "bska1"],
    "�����Ȩ��": ["�ͻس���ʺ����ϲ���", "������С��ֵ��ϲ���", "bska1"],
    "��ϸ�������": [1, 1, 0.1],
    "��ͬ���Լ������ʱ��": 0,
    "�µ�Ĭ��˵��": "Ĭ��/���/����",
    "�µ�ģʽ": "Ĭ��",
    "�µ�ֵ": 1000,
    "ʱ������": "ʱ������********",
    "����ʱ���": 4,
    "���׿�ʼʱ��": 9,
    "���׽���ʱ��": 15,
    "�Ƿ�μӼ��Ͼ���": "��",
    "��ʼ���׷���": 0,
    "�Ƿ�����ʱid��¼": "��",
    "����΢����Ϣ": "��",
    "���Ͷ�����Ϣ": "��"
}


# ��¼��ʱid,����ѭ����û���õĵ���
class a:
    pass


a = a()
a.log_id = []


def init(c):
    # �˻�
    c.account = text['�˻�']
    # �˻�����
    c.account_type = text['�˻�����']
    c.url = text['������']
    c.port = text['�˿�']
    print('С���������ṩ����֧��************������{} �˿�{}'.format(c.url, c.port))
    # ��ʱģʽ
    # c.run_time("update_all_data","1nDay","2024-07-25 09:45:00")
    # c.run_time("update_all_data","1nDay","2024-07-25 14:45:00")
    # ѭ��ģʽ3��
    c.run_time("update_all_data", "1nSecond", "2024-07-25 13:20:00")
    # c.run_time("tarder_test","3nSecond","2024-07-25 13:20:00")
    print(get_account(c, c.account, c.account_type))
    print(get_position(c, c.account, c.account_type))


# print(update_all_data(c))
def handlebar(c):
    pass


def tarder_test(c):
    print('���ײ���***************')
    stock = '513100.SH'
    amount = 100
    maker = '���ײ���'
    passorder(23, 1101, c.account, stock, 5, 0, amount, maker, 1, maker, c)


def get_del_buy_sell_data(c, name='����1', password='123456'):
    '''
    ���������ݻ�ȡԭʼ����
    '''
    test = text['�Ƿ�������']
    url = text['������']
    port = text['�˿�']
    now_date = str(datetime.now())[:10]
    xg_data = xg_jq_data(url=url, port=port, password=password)
    info = xg_data.get_user_data(data_type='�û���Ϣ')
    df = xg_data.get_user_data(data_type='ʵʱ����')
    # print('�û���Ϣ�Ѿ���ȡ')
    # print(info)
    if df.shape[0] > 0:
        stats = df['����״̬'].tolist()[-1]
        if stats == True:
            df['֤ȯ����'] = df['��Ʊ����'].apply(
                lambda x: str(x).split('.')[0] + '.SH' if str(x).split('.')[-1] == 'XSHG' else str(x).split('.')[
                                                                                                   0] + '.SZ')
            df['���ݳ���'] = df['֤ȯ����'].apply(lambda x: len(str(x)))
            df['�������ʱ��'] = df['�������ʱ��'].apply(lambda x: str(x)[:10])
            df = df[df['���ݳ���'] >= 6]
            df['��������'] = df['����'].apply(lambda x: 'buy' if x == True else 'sell')
            df = df.drop_duplicates(subset=['��Ʊ����', '�µ�����', '����', '���'], keep='last')
            df['�������'] = str(name)
            df['�����Ȩ��'] = str(password)
            df['��������'] = str('�ۿ����')
            df['����ID'] = df['����ID'].astype(str)
            df['֤ȯ����'] = df['֤ȯ����'].apply(lambda x: str(x))
            df['Ͷ�ʱ�ע'] = df['�����Ȩ��'] + ',' + df['����ID']
            if test == '��':
                print('��������ģʽ,ʵ�̼ǵùر�')
                df = df
            else:
                df = df[df['�������ʱ��'] == now_date]
        else:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()
    if df.shape[0] > 0:
        print('��� {} ������Ȩ�� {} {}�����и�������*********************'.format(name, password, now_date))
    # print(df)
    else:
        print('��� {} ������Ȩ�� {} {}����û�и�������*********************'.format(name, password, now_date))
    return df


# ����΢����Ϣ
def send_wx_message(message, item='��ͬQMTʵ��'):
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
    # ���Ը�����Ҫ�鿴��Ӧ��״̬�롢���ݵ���Ϣ
    # print(response.status_code)
    # print(response.text)


def seed_dingding(message='�������׳ɹ�',
                  access_token_list=['2615283ba0ab84900235e4ecc262671a0c79f3fb13acdf35f22cfd4b0407295f']):
    access_token = random.choice(access_token_list)
    url = 'https://oapi.dingtalk.com/robot/send?access_token={}'.format(access_token)
    headers = {'Content-Type': 'application/json;charset=utf-8'}
    data = {
        "msgtype": "text",  # ������Ϣ����Ϊ�ı�
        "at": {
            # "atMobiles": reminders,
            "isAtAll": False,  # ��@������
        },
        "text": {
            "content": '��ͬQMT����֪ͨ\n' + message,  # ��Ϣ����
        }
    }
    r = requests.post(url, data=json.dumps(data), headers=headers)
    text = r.json()
    errmsg = text['errmsg']
    if errmsg == 'ok':
        print('�������ͳɹ�')
        return text
    else:
        print(text)
        return text


def get_trader_data(c, name='����', password='123456', zh_ratio=0.1):
    '''
    ��ȡ��������
    ��ϵĸ�������
    '''
    test = text['�Ƿ�������']
    adjust_ratio = text['�˻���������']
    is_open_id_log = text['�Ƿ�����ʱid��¼']
    # ��ȡ��������
    df = get_del_buy_sell_data(c, name=name, password=password)
    try:
        df['֤ȯ����'] = df['֤ȯ����'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
    except:
        pass
    trader_log = get_order(c, c.account, c.account_type)
    now_date = str(datetime.now())[:10]
    # �޳������ϵ�
    not_list = [54, 57]
    if trader_log.shape[0] > 0:
        trader_log['����'] = trader_log['ί��״̬'].apply(lambda x: '��' if x in not_list else '����')
        trader_log = trader_log[trader_log['����'] == '����']
    else:
        trader_log = trader_log
    if trader_log.shape[0] > 0:
        trader_log['֤ȯ����'] = trader_log['֤ȯ����'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
        trader_log['�����Ȩ��'] = trader_log['Ͷ�ʱ�ע'].apply(lambda x: str(x).split(',')[0])
        trader_log['����ID'] = trader_log['Ͷ�ʱ�ע'].apply(lambda x: str(x).split(',')[-1])
        trader_log['����ID'] = trader_log['����ID'].astype(str)
        if test == '��':
            trader_log = trader_log
        else:
            trader_log = trader_log
        trader_log['�����Ȩ��'] = trader_log['�����Ȩ��'].astype(str)
        trader_log_1 = trader_log[trader_log['�����Ȩ��'] == password]
        if trader_log_1.shape[0] > 0:
            trader_id_list = trader_log_1['����ID'].tolist()
        else:
            trader_id_list = []
    else:
        trader_id_list = []
    if is_open_id_log == '��':
        for trader_id in a.log_id:
            trader_id_list.append(trader_id)
    else:
        pass
    trader_id_list = list(set(trader_id_list))

    if df.shape[0] > 0:
        df['�����Ȩ��'] = df['�����Ȩ��'].astype(str)
        # df['����ID ']=df['����ID'].astype(str)
        df = df[df['�����Ȩ��'] == password]
        if df.shape[0] > 0:
            df['�˻���������'] = adjust_ratio
            df['��ϸ�������'] = zh_ratio
            df['���׼��'] = df['����ID'].apply(lambda x: '�Ѿ�����' if x in trader_id_list else 'û�н���')
            df = df[df['���׼��'] == 'û�н���']
            amount_list = []
            if df.shape[0] > 0:
                for stock, amount, trader_type in zip(df['֤ȯ����'].tolist(), df['�µ�����'].tolist(),
                                                      df['��������'].tolist()):
                    try:
                        price = get_price(c, stock=stock)
                        test = text['�Ƿ�������']
                        test_amount = text['��������']
                        down_type = text['�µ�ģʽ']
                        down_value = text['�µ�ֵ']
                        send_wx_msg = text['����΢����Ϣ']
                        send_dd_msg = text['���Ͷ�����Ϣ']
                        if test == '��':
                            value = test_amount * price
                        else:
                            if down_type == 'Ĭ��':
                                value = price * amount * adjust_ratio * zh_ratio
                            elif down_type == '����':
                                value = price * down_value * adjust_ratio * zh_ratio
                            elif down_type == '���':
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

                                msg = f'��ǰʱ��--{str(datetime.now())[:19]}\n�����Ʊ--{stock}\n����--{amount}\n����--{price}\n�ܼ�--{price * amount}'
                                if send_wx_msg == '��':
                                    send_wx_message(message=msg)
                                if send_dd_msg == '��':
                                    seed_dingding(message=msg)
                            except Exception as e:
                                print('���{} �����Ȩ��{} {}�������������û���ʽ�'.format(name, password, stock))
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

                                msg = f'��ǰʱ��--{str(datetime.now())[:19]}\n������Ʊ--{stock}\n����--{amount}\n����--{price}\n�ܼ�--{price * amount}'
                                if send_wx_msg == '��':
                                    send_wx_message(message=msg)
                                if send_dd_msg == '��':
                                    seed_dingding(message=msg)
                            except Exception as e:

                                print('���{} �����Ȩ��{} {}�������������û�гֹ�'.format(name, password, stock))
                                amount_list.append(0)

                        else:
                            print('���{} �����Ȩ��{} {}δ֪�Ľ�������'.format(name, password, stock))

                    except Exception as e:
                        print(e, stock, '������*************')
                        amount_list.append(0)

                df['����'] = amount_list
                not_trader = df[df['����'] <= 0]
                # ����Ϊ0�Ĳ������µ���¼
                df = df[df['����'] >= 10]
                # df=df[df['����']>=0]
                print('�µ���Ʊ��*************')
                print(df)
                print(
                    '�µ�����Ϊ0�ı�Ŀ���û�гֹ�,�����˻�û���ʽ�ȴ��´γɽ�########################################################')
                print(not_trader)
                trader_log = pd.concat([trader_log, df], ignore_index=True)
                trader_log = trader_log.drop_duplicates(subset=['�������ʱ��', '����ID', '�����Ȩ��', '�������'],
                                                        keep='last')
            else:
                # print('{}���û����Ҫ�µ���Ʊ******************'.format(name))
                df = pd.DataFrame()
        else:
            print('{}û��������*************'.format(name))
            df = pd.DataFrame()

    else:
        # print('{}���׹�Ʊ��û������*************'.format(name))
        df = pd.DataFrame()
    return df


def start_trader_on(c, name='����1', password='123456', zh_ratio=0.1):
    '''
    ��ʼ�µ�
    '''
    is_open_id_log = text['�Ƿ�����ʱid��¼']
    df = get_trader_data(c, name, password=password, zh_ratio=zh_ratio)
    try:
        df['֤ȯ����'] = df['֤ȯ����'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
    except:
        pass
    if df.shape[0] > 0:
        df['֤ȯ����'] = df['֤ȯ����'].astype(str)
        # print(df['֤ȯ����'])
        df['֤ȯ����'] = df['֤ȯ����'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
        # ��������
        sell_df = df[df['��������'] == 'sell']
        if sell_df.shape[0] > 0:
            for stock, amount, maker, in zip(sell_df['֤ȯ����'].tolist(), sell_df['����'].tolist(),
                                             sell_df['Ͷ�ʱ�ע'].tolist()):
                try:

                    price = get_price(c, stock)
                    passorder(24, 1101, c.account, str(stock), 5, 0, int(amount), str(maker), 1, str(maker), c)
                    print('���{} �������{} ����{} �۸�{}'.format(name, stock, amount, price))
                    if is_open_id_log == '��':
                        trader_id = str(maker).split(',')[-1]
                        a.log_id.append(trader_id)
                    else:
                        pass

                except Exception as e:
                    print(e)
                    print('���{} {}����������'.format(name, stock))
        else:
            print('{}���û�з��ϵ��ε���������'.format(name))
        buy_df = df[df['��������'] == 'buy']
        if buy_df.shape[0] > 0:
            for stock, amount, maker, in zip(buy_df['֤ȯ����'].tolist(), buy_df['����'].tolist(),
                                             buy_df['Ͷ�ʱ�ע'].tolist()):
                try:

                    price = get_price(c, stock)
                    passorder(23, 1101, c.account, str(stock), 5, 0, int(amount), str(maker), 1, str(maker), c)
                    # passorder(23, 1101, c.account, stock, 5, 0, int(amount), '',1,'',c)
                    print('���{} ������{} ����{} �۸�{}'.format(name, stock, amount, price))
                    if is_open_id_log == '��':
                        trader_id = str(maker).split(',')[-1]
                        a.log_id.append(trader_id)
                    else:
                        pass

                except Exception as e:
                    print(e)
                    print('���{} {}����������'.format(name, stock))

        else:
            print('{}���û�з��ϵ��ε���������'.format(name))
    # else:
    #     print('{}���û�з��ϵ�������'.format(name))


# print(a.log_id)
def update_all_data(c):
    '''
    ���²�������
    '''
    if check_is_trader_date_1():
        name_list = text['�������']
        password_list = text['�����Ȩ��']
        ratio_list = text['��ϸ�������']
        update_time = text['��ͬ���Լ������ʱ��']
        for name, password, ratio in zip(name_list, password_list, ratio_list):
            print('������������������---{}---�ָ�������������������'.format(name))
            start_trader_on(c, name=name, password=password, zh_ratio=ratio)
            time.sleep(update_time * 60)
    else:
        print('����{} Ŀǰ���ǽ���ʱ��***************'.format(datetime.now()))
        time.sleep(30)


def order_stock_value(c, accountid, datatype, stock, value, trader_type):
    '''
    ��ֵ�µ�����
    '''
    price = get_price(c, stock)
    hold_stock = get_position(c, accountid, datatype)
    if hold_stock.shape[0] > 0:
        hold_stock = hold_stock[hold_stock['�ֲ���'] >= 10]
        if hold_stock.shape[0] > 0:
            hold_df = hold_stock[hold_stock['֤ȯ����'] == stock]
            if hold_df.shape[0] > 0:
                hold_amount = hold_df['�ֲ���'].tolist()[-1]
                av_amount = hold_df['��������'].tolist()[-1]
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
    av_cash = account['���ý��']
    amount = value / price
    if str(stock)[:2] in ['11', '12']:
        amount = int(amount / 10) * 10
    else:
        amount = int(amount / 100) * 100
    if trader_type == 'buy':
        if av_cash >= value and amount >= 10:
            print('����µ������ʽ�{}����������{} ����{} �۸�{} ����{}'.format(av_cash, value, stock, price, amount))
            return 'buy', amount, price
        else:
            print(
                '����µ������ʽ�{}С��������{} ������{} �۸�{} ����{}'.format(av_cash, value, stock, price, amount))
            return '', '', price
    elif trader_type == 'sell':
        if av_amount >= amount and amount >= 10:
            print('����µ� ��������{} ��������{} ������������{} ����{} �۸�{} ����{}'.format(hold_amount, av_amount,
                                                                                              amount, stock, price,
                                                                                              amount))
            return 'sell', amount, price
        elif av_amount < amount and av_amount >= 10:
            print(
                '����µ� ��������{} ��������{} С����������{}��������������10 ����{} �۸�{} ����{}'.format(hold_amount,
                                                                                                           av_amount,
                                                                                                           amount,
                                                                                                           stock, price,
                                                                                                           amount))
            return 'sell', amount, price
        else:
            print('����µ� ��������{} ��������{} С����������{}��������{} �۸�{} ����{}'.format(hold_amount, av_amount,
                                                                                                amount, stock, price,
                                                                                                amount))
            return 'sell', amount, price
    else:
        print('����µ�δ֪�Ľ�������{}'.format(stock))
        return '', amount, price


def buy(c, stock, price, amount, name):
    '''
    ���뺯��
    '''
    passorder(23, 1101, c.account, str(stock), 5, 0, amount, name, 1, name, c)


def sell(c, stock, price, amount, name):
    '''
    �Զ�����������
    '''
    passorder(24, 1101, c.account, str(stock), 5, 0, amount, name, 1, name, c)


def get_price(c, stock):
    '''
    ��ȡ���¼۸�
    '''
    tick = c.get_full_tick(stock_code=[stock])
    tick = tick[stock]
    price = tick['lastPrice']
    return price


def adjust_amount(c, stock='', amount=''):
    '''
    ��������
    '''
    if stock[:3] in ['110', '113', '123', '127', '128', '111'] or stock[:2] in ['11', '12']:
        amount = math.floor(amount / 10) * 10
    else:
        amount = int(round(amount, -2))
    return amount


def check_is_trader_date_1():
    '''
    ����ǲ��ǽ���ʱ��
    '''
    trader_time = text['����ʱ���']
    start_date = text['���׿�ʼʱ��']
    end_date = text['���׽���ʱ��']
    start_mi = text['��ʼ���׷���']
    jhjj = text['�Ƿ�μӼ��Ͼ���']
    if jhjj == '��':
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
        print('��ĩ')
        return False


# ��ȡ�˻���Ȩ��m_dBalance
def get_account(c, accountid, datatype):
    '''
    ��ȡ�˻�����
    '''
    accounts = get_trade_detail_data(accountid, datatype, 'account')
    result = {}
    for dt in accounts:
        result['���ʲ�'] = dt.m_dBalance
        result['���ʲ�'] = dt.m_dAssureAsset
        result['����ֵ'] = dt.m_dInstrumentValue
        result['�ܸ�ծ'] = dt.m_dTotalDebit
        result['���ý��'] = dt.m_dAvailable
        result['ӯ��'] = dt.m_dPositionProfit
    return result


# ��ȡ�ֲ���Ϣ{code.market:����}
def get_position(c, accountid, datatype):
    '''
    ��ȡ�ֹ�����
    '''
    positions = get_trade_detail_data(accountid, datatype, 'position')
    data = pd.DataFrame()
    print('�ֹ�����{}'.format(len(positions)))
    if len(positions) > 0:
        df = pd.DataFrame()
        try:
            for dt in positions:
                df['��Ʊ����'] = [dt.m_strInstrumentID]
                df['�г�����'] = [dt.m_strExchangeID]
                df['֤ȯ����'] = df['��Ʊ����'] + '.' + df['�г�����']
                df['֤ȯ����'] = [dt.m_strInstrumentName]
                df['�ֲ���'] = [dt.m_nVolume]
                df['��������'] = [dt.m_nCanUseVolume]
                df['�ɱ���'] = [dt.m_dOpenPrice]
                df['��ֵ'] = [dt.m_dInstrumentValue]
                df['�ֲֳɱ�'] = [dt.m_dPositionCost]
                df['ӯ��'] = [dt.m_dPositionProfit]
                data = pd.concat([data, df], ignore_index=True)

        except Exception as e:
            print('��ȡ�ֹɸ����Ʊ��������')
            data = pd.DataFrame()
    else:
        data = pd.DataFrame()
    return data


def get_order(c, accountid, datatype):
    '''
    ��ȡί��
    '''
    data = pd.DataFrame()
    orders = get_trade_detail_data(accountid, datatype, 'order')
    print('ί������{}'.format(len(orders)))
    if len(orders) > 0:
        df = pd.DataFrame()
        for o in orders:
            df['��Ʊ����'] = [o.m_strInstrumentID]
            df['�г�����'] = [o.m_strExchangeID]
            df['֤ȯ����'] = df['��Ʊ����'] + '.' + df['�г�����']
            df['��������'] = [o.m_nOffsetFlag]
            df['ί������'] = [o.m_nVolumeTotalOriginal]
            df['�ɽ�����'] = [o.m_dTradedPrice]
            df['�ɽ�����'] = [o.m_nVolumeTraded]
            df['�ɽ����'] = [o.m_dTradeAmount]
            df['Ͷ�ʱ�ע'] = [o.m_strRemark]
            df['ί��״̬'] = [o.m_nOrderStatus]
            data = pd.concat([data, df], ignore_index=True)
    else:
        data = pd.DataFrame()
    return data


def get_deal(c, accountid, datatype):
    '''
    ��ȡ�ɽ�
    '''
    data = pd.DataFrame()
    deals = get_trade_detail_data(account, 'stock', 'deal')
    print('�ɽ�����{}'.format(len(deals)))
    if len(deals):
        df = pd.DataFrame()
        for dt in deals:
            df['��Ʊ����'] = [dt.m_strInstrumentID]
            df['�г�����'] = [dt.m_strExchangeID]
            df['֤ȯ����'] = df['��Ʊ����'] + '.' + df['�г�����']
            df['֤ȯ����'] = [dt.m_strInstrumentName]
            df['��������'] = [dt.m_nOffsetFlag]
            df['�ɽ��۸�'] = [dt.m_dPrice]
            df['�ɽ�����'] = [dt.m_nVolume]
            df['�ɽ����'] = [dt.m_dTradeAmount]
            data = pd.concat([data, df], ignore_index=True)
    else:
        data = pd.DataFrame()


class xg_jq_data:
    def __init__(self, url='http://124.220.32.224', port=8025, password='123456'):
        '''
        ��ȡ����������
        '''
        self.url = url
        self.port = port
        self.password = password

    def get_user_data(self, data_type='�û���Ϣ'):
        '''
        ��ȡʹ�õ�����
        data_type='�û���Ϣ','ʵʱ����',��ʷ����','���ʵʱ����','�����ʷ����'
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
                        "value": "\n               {'״̬': 'held', '�������ʱ��': 'datetime.datetime(2024, 4, 23, 9, 30)', '����': 'False', '�µ�����': '9400', '�Ѿ��ɽ�': '9400', '��Ʊ����': '001.XSHE', '����ID': '1732208241', 'ƽ���ɽ��۸�': '10.5', '�ֲֳɱ�': '10.59', '���': 'long', '���׷���': '128.31'}\n                "},
                       {"id": "joinquant_trader_run", "property": "value", "value": "����"},
                       {"id": "joinquant_trader_down_data", "property": "value", "value": "����������"}],
            "changedPropIds": ["joinquant_trader_run.value"], "parsedChangedPropsIds": ["joinquant_trader_run.value"]}
        res = requests.post(url=url, data=json.dumps(data), headers=headers)
        text = res.json()
        df = pd.DataFrame(text['response']['joinquant_trader_table']['data'])
        return df
