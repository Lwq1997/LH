# encoding:gbk
'''
С���ۿ��׳ɽ�ϵͳ���ٷ�����2
���ߣ�С������
΢��:xg_quant
ʱ�䣺20250507
�Ż�����
1:���׼�鹦��,�Զ�����
2:5���Ӳ��ɽ������������µ�
3���Ż��˽����㷨������ϸ��
�̳� https://gitee.com/li-xingguo11111/joinquant_trader_miniqmt
��qmt https://gitee.com/li-xingguo11111/joinquant_trader_bigqmt
��Ƶʹ��ѭ��ģʽ����Ƶʹ�ö�ʱ
��ֵ	����
-1	��Ч(ֻ����algo_passorder������)
0	��5��
1	��4��
2	��3��
3	��2��
4	��1��
5	���¼�
6	��1��
7	��2��(��ϲ�֧��)
8	��3��(��ϲ�֧��)
9	��4��(��ϲ�֧��)
10	��5��(��ϲ�֧��)
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
    "�˻�֧��������ȯ": "�˻�֧��������ȯ,�˻�����STOCK/CREDIT",
    "�˻�����": "STOCK",
    "�ۿ����": "����ԭ��",
    "����������": "��������������",
    "�Ƿ���id��¼": "��",
    # "������": "http://115.175.23.7",
    # "�˿�": "2000",
    # "����˵��": "�������Ծ���ѡ����ʷ���ײ���������ѡ����������",
    # "�Ƿ�������": "��",
    # "��������": 100,
    # "��������": "��������***********",
    # "�˻���������": 0.5,
    # "������ö��Ÿ���": "������ö��Ÿ���********",
    # "�������": ["8n", "һ����", "xjy_001"],
    # "�����Ȩ��": ["8n", "789789", "xjy_001"],
    # "��ϸ�������": [1, 1, 1],
    "�µ�Ĭ��˵��": "Ĭ��/���/����",
    "�µ�ģʽ": "Ĭ��",
    "�µ�ֵ": 1000,
    "ʱ������": "ʱ������********",
    "����ʱ���": 8,
    "���׿�ʼʱ��": 8,
    "���׽���ʱ��": 24,
    "�Ƿ�μӼ��Ͼ���": "��",
    "��ʼ���׷���": 0,
    "����΢����Ϣ": "��",
    "���Ͷ�����Ϣ": "��"
}
data = [
    {
        "������": "http://server.588gs.cn",
        "�˿�": "2000",
        "����˵��": "�������Ծ���ѡ����ʷ���ײ���������ѡ����������",
        "�Ƿ�������": "��",
        "��������": 100,
        "��������": "��������***********",
        "�˻���������": 1,
        "������ö��Ÿ���": "������ö��Ÿ���********",
        "�������": ["�ͻس���ʺ����ϲ���", "������С��ֵ��ϲ���", "xjy_bska"],
        "�����Ȩ��": ["�ͻس���ʺ����ϲ���", "������С��ֵ��ϲ���", "xjy_bska"],
        "��ϸ�������": [1, 1, 1],
        "��ͬ���Լ������ʱ��": 0,
        "����۸����": 4,
        "�����۸����": 6,
        "������˵��": "����������ı�ģ���������Ҳ��������",
        "������": []
    },
    {
        "������": "http://106.54.211.231",
        "�˿�": "3333",
        "����˵��": "�������Ծ���ѡ����ʷ���ײ���������ѡ����������",
        "�Ƿ�������": "��",
        "��������": 100,
        "��������": "��������***********",
        "�˻���������": 1,
        "������ö��Ÿ���": "������ö��Ÿ���********",
        "�������": ["�ͻس���ʺ����ϲ���", "������С��ֵ��ϲ���", "xjy_bska"],
        "�����Ȩ��": ["�ͻس���ʺ����ϲ���", "������С��ֵ��ϲ���", "xjy_bska"],
        "��ϸ�������": [1, 1, 1],
        "��ͬ���Լ������ʱ��": 0,
        "����۸����": 4,
        "�����۸����": 6,
        "������˵��": "����������ı�ģ���������Ҳ��������",
        "������": []
    },
    # ����������
    {
        "������": "http://server.588gs.cn",
        "�˿�": "2000",
        "����˵��": "�������Ծ���ѡ����ʷ���ײ���������ѡ����������",
        "�Ƿ�������": "��",
        "��������": 100,
        "��������": "��������***********",
        "�˻���������": 1,
        "������ö��Ÿ���": "������ö��Ÿ���********",
        "�������": ["���ST����", "1n2_bska", "8n_bska"],
        "�����Ȩ��": ["���ST����", "1n2_bska", "8n_bska"],
        "��ϸ�������": [1, 0.5, 0.5],
        "��ͬ���Լ������ʱ��": 0,
        "����۸����": 3,
        "�����۸����": 7,
        "������˵��": "����������ı�ģ���������Ҳ��������",
        "������": []
    },
    {
        "������": "http://106.54.211.231",
        "�˿�": "3333",
        "����˵��": "�������Ծ���ѡ����ʷ���ײ���������ѡ����������",
        "�Ƿ�������": "��",
        "��������": 100,
        "��������": "��������***********",
        "�˻���������": 1,
        "������ö��Ÿ���": "������ö��Ÿ���********",
        "�������": ["���ST����", "1n2_bska", "8n_bska"],
        "�����Ȩ��": ["���ST����", "1n2_bska", "8n_bska"],
        "��ϸ�������": [1, 0.5, 0.5],
        "��ͬ���Լ������ʱ��": 0,
        "����۸����": 3,
        "�����۸����": 7,
        "������˵��": "����������ı�ģ���������Ҳ��������",
        "������": []
    }
]


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
    if c.account_type == 'stock' or c.account_type == 'STOCK':
        c.buy_code = 23
        c.sell_code = 24
    else:
        # ������ȯ
        c.buy_code = 33
        c.sell_code = 34
    for item in data:
        url = item['������']
        port = item['�˿�']
        print('\n С���������ṩ����֧��************��������{}�� �˿ڡ�{}��'.format(url, port))
    # ��ʱģʽ
    # c.run_time("update_all_data","1nDay","2024-07-25 09:45:00")
    # c.run_time("update_all_data","1nDay","2024-07-25 14:45:00")
    # ѭ��ģʽ3��
    c.run_time("update_all_data", "60nSecond", "2024-07-25 13:20:00")
    # c.run_time("tarder_test","3nSecond","2024-07-25 13:20:00")
    # ���׼�麯��1����һ��
    c.run_time("run_check_trader_func", "60nSecond", "2024-07-25 13:20:00")
    # �����������µ�5����һ��
    c.run_time("run_order_trader_func", "300nSecond", "2024-07-25 13:20:00")
    print(get_account(c, c.account, c.account_type))
    print(get_position(c, c.account, c.account_type))


def handlebar(c):
    pass


def tarder_test(c):
    print('���ײ���***************')
    stock = '513100.SH'
    amount = 100
    maker = '���ײ���'
    passorder(23, 1101, c.account, stock, 5, 0, amount, maker, 1, maker, c)


def get_del_buy_sell_data(c, name='����1', password='123456', item=None):
    '''
    ���������ݻ�ȡԭʼ����
    '''
    test = item['�Ƿ�������']
    url = item['������']
    port = item['�˿�']
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
            df['Ͷ�ʱ�ע'] = df['�����Ȩ��'] + ',' + df['֤ȯ����'] + ',' + df['����ID']
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
        print('\n ��� ��{}�� ������Ȩ�� ��{}����{}�������и�������*********************'.format(name, password, now_date))
    # print(df)
    else:
        print('\n ��� ��{}��������Ȩ�롾{}����{}������û�и�������*********************'.format(name, password, now_date))
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


def get_trader_data(c, name='����', password='123456', zh_ratio=0.1, item=None):
    '''
    ��ȡ��������
    ��ϵĸ�������
    '''
    test = item['�Ƿ�������']
    adjust_ratio = item['�˻���������']
    # ��ȡ��������
    df = get_del_buy_sell_data(c, name=name, password=password, item=item)
    try:
        df['֤ȯ����'] = df['֤ȯ����'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
    except:
        pass
    trader_log = get_order(c, c.account, c.account_type)
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
                        test = item['�Ƿ�������']
                        test_amount = item['��������']
                        down_type = text['�µ�ģʽ']
                        down_value = text['�µ�ֵ']
                        send_wx_msg = text['����΢����Ϣ']
                        send_dd_msg = text['���Ͷ�����Ϣ']
                        if test == '��':
                            value = price * test_amount * adjust_ratio * zh_ratio
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

                                msg = f'��ǰʱ��--{str(datetime.now())[:19]}\n���--{name}\n�����Ʊ--{stock}\n����--{amount}\n����--{price}\n�ܼ�--{price * amount}'
                                if send_wx_msg == '��':
                                    send_wx_message(message=msg)
                                if send_dd_msg == '��':
                                    seed_dingding(message=msg)
                            except Exception as e:
                                print('��ϡ�{}�� �����Ȩ�롾{}����{}���������������û���ʽ�'.format(name, password, stock))
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

                                msg = f'��ǰʱ��--{str(datetime.now())[:19]}\n���--{name}\n������Ʊ--{stock}\n����--{amount}\n����--{price}\n�ܼ�--{price * amount}'
                                if send_wx_msg == '��':
                                    send_wx_message(message=msg)
                                if send_dd_msg == '��':
                                    seed_dingding(message=msg)
                            except Exception as e:

                                print('��ϡ�{}�� �����Ȩ�롾{}�� ��{}���������������û�гֹ�'.format(name, password, stock))
                                amount_list.append(0)

                        else:
                            print('��ϡ�{}�� �����Ȩ�롾{}�� ��{}��δ֪�Ľ�������'.format(name, password, stock))

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
                # print('��{}�����û����Ҫ�µ���Ʊ******************'.format(name))
                df = pd.DataFrame()
        else:
            print('��{}��û��������*************'.format(name))
            df = pd.DataFrame()

    else:
        # print('��{}�����׹�Ʊ��û������*************'.format(name))
        df = pd.DataFrame()
    return df


def check_is_sell(c, accountid, datatype, stock='', amount=0):
    '''
    ����Ƿ��������
    '''
    position = get_position(c, accountid, datatype)
    if position.shape[0] > 0:
        position = position[position['֤ȯ����'] == stock]
        if position.shape[0] > 0:
            position = position[position['�ֲ���'] >= 0]
            if position.shape[0] > 0:
                av_amount = position['��������'].tolist()[-1]
                if av_amount >= 10:
                    return True
                else:
                    print('��{}�� ������������������{}�� С������������{}��'.format(stock, av_amount, amount))
                    return False
            else:
                print('��{}�� ����������û�гֹ�'.format(stock))
                return False
        else:
            print('��{}�� ����������û�гֹ�'.format(stock))
            return False
    else:
        print('��{}�� �˻��ղ�'.format(stock))
        return False


def check_is_buy(c, accountid, datatype, stock='', amount=0, price=0):
    '''
    ����Ƿ��������
    '''
    account = get_account(c, accountid, datatype)
    # ����ʹ�õ��ֽ�
    av_cash = account['���ý��']
    value = amount * price
    if av_cash >= value:
        return True
    else:
        print('��{}�� �˻������ʽ�{}�� ��������۸�{}�� ������{}�� ���'.format(stock, av_cash, price, amount))
        return False


def start_trader_on(c, name='����1', password='123456', zh_ratio=0.1, item=None):
    '''
    ��ʼ�µ�
    '''
    is_open_id_log = text['�Ƿ���id��¼']
    del_trader_list = item['������']
    sell_price_code = item['�����۸����']
    buy_price_code = item['����۸����']

    hold_stock = get_position(c, c.account, c.account_type)
    if hold_stock.shape[0] > 0:
        hold_stock = hold_stock[hold_stock['�ֲ���'] >= 10]
        if hold_stock.shape[0] > 0:
            hold_stock_list = hold_stock['֤ȯ����'].tolist()
        else:
            hold_stock_list = []
    else:
        hold_stock_list = []
    df = get_trader_data(c, name, password=password, zh_ratio=zh_ratio, item=item)
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
        if len(hold_stock_list) > 0:
            av_amount_dict = dict(zip(hold_stock['֤ȯ����'].tolist(), hold_stock['��������'].tolist()))
        else:
            av_amount_dict = {}
        if sell_df.shape[0] > 0:
            sell_df['��������'] = sell_df['֤ȯ����'].apply(lambda x: av_amount_dict.get(x, 0))
            for stock, av_amount, amount, maker, in zip(sell_df['֤ȯ����'].tolist(),
                                                        sell_df['��������'].tolist(),
                                                        sell_df['����'].tolist(),
                                                        sell_df['Ͷ�ʱ�ע'].tolist()):
                if stock not in del_trader_list:
                    if maker not in a.log_id:
                        # print('{} ��Ĳ��ں���������'.format(stock))
                        amount = amount if av_amount >= amount else av_amount
                        if amount >= 10:
                            try:
                                price = get_price(c, stock)
                                if check_is_sell(c, c.account, c.account_type, stock=stock, amount=amount):
                                    passorder(c.sell_code, 1101, c.account, str(stock), sell_price_code, 0,
                                              int(amount), str(maker), 1, str(maker), c)
                                    print('��ϡ�{}�� ������ġ�{}�� ������{}�� �۸�{}��'.format(name, stock, amount, price))
                                    if is_open_id_log == '��':
                                        a.log_id.append(maker)
                                    else:
                                        pass
                                else:
                                    print('��ϡ�{}�� ��{}����������'.format(name, stock))
                            except Exception as e:
                                print(e)
                                print('��ϡ�{}�� ��{}������������'.format(name, stock))
                        else:
                            print("��{}�� ��{}�� �������˿���������{}��".format(name, stock, av_amount))
                    else:
                        print('��{}�� ��{}�� ��id��¼���������ȴ�����ȷ�ϼ��'.format(name, stock))
                else:
                    print('��{}�� ��{}�� ����ں�����������'.format(name, stock))
        else:
            print('��{}�����û�з��ϵ��ε���������'.format(name))
        buy_df = df[df['��������'] == 'buy']
        if buy_df.shape[0] > 0:
            for stock, amount, maker, in zip(buy_df['֤ȯ����'].tolist(), buy_df['����'].tolist(),
                                             buy_df['Ͷ�ʱ�ע'].tolist()):
                if stock not in del_trader_list:
                    # print('��{}�� ��Ĳ��ں���������'.format(stock))
                    if maker not in a.log_id:
                        try:
                            price = get_price(c, stock)
                            if check_is_buy(c, c.account, c.account_type, stock=stock, amount=amount, price=price):
                                passorder(c.buy_code, 1101, c.account, str(stock), buy_price_code, 0, int(amount),
                                          str(maker), 1, str(maker), c)
                                # passorder(23, 1101, c.account, stock, 5, 0, int(amount), '',1,'',c)
                                print('��ϡ�{}�� �����ġ�{}�� ������{}�� �۸�{}��'.format(name, stock, amount, price))
                                if c.is_open_id_log == '��':
                                    a.log_id.append(maker)
                                else:
                                    pass
                            else:
                                print('��ϡ�{}�� ��{}����������'.format(name, stock))
                        except Exception as e:
                            print(e)
                            print('��ϡ�{}�� ��{}������������'.format(name, stock))
                    else:
                        print('��{}�� ��{}�� ��id��¼�����룬�ȴ�����ȷ�ϼ��'.format(name, stock))
                else:
                    print('��{}�� ����ں�����������'.format(stock))

        else:
            print('��{}�����û�з��ϵ��ε���������'.format(name))
    # else:
    #     print('��{}�����û�з��ϵ�������'.format(name))


# print(a.log_id)
def update_all_data(c):
    '''
    ���²�������
    '''
    if check_is_trader_date_1():
        for item in data:
            name_list = item['�������']
            password_list = item['�����Ȩ��']
            ratio_list = item['��ϸ�������']
            update_time = item['��ͬ���Լ������ʱ��']
            for name, password, ratio in zip(name_list, password_list, ratio_list):
                print('������������������---��{}��-----IP---��{}��----�˿�---��{}��---�ָ�������������������'.format(name, item['������'],
                                                                                          item['�˿�']))
                start_trader_on(c, name=name, password=password, zh_ratio=ratio, item=item)
                time.sleep(update_time * 60)
    else:
        print('������{}�� Ŀǰ���ǽ���ʱ��***************'.format(datetime.now()))
        time.sleep(30)


def run_check_trader_func(c):
    '''
    ��齻���µ������û���µ��ľͲ���
    '''
    trader_log = get_order(c, c.account, c.account_type)
    # �޳������ϵ�
    not_list = [54, 57]
    if trader_log.shape[0] > 0:
        trader_log['����'] = trader_log['ί��״̬'].apply(lambda x: '��' if x in not_list else '����')
        trader_log = trader_log[trader_log['����'] == '����']

    if trader_log.shape[0] > 0:
        trader_log['֤ȯ����'] = trader_log['֤ȯ����'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
        trader_log['�����Ȩ��'] = trader_log['Ͷ�ʱ�ע'].apply(lambda x: str(x).split(',')[0])
        trader_log['����ID'] = trader_log['Ͷ�ʱ�ע'].apply(lambda x: str(x).split(',')[-1])
        trader_log['����ID'] = trader_log['����ID'].astype(str)
        maker_list = trader_log['Ͷ�ʱ�ע'].tolist()
    else:
        maker_list = []
    # ����id��¼a
    if len(a.log_id) > 0:
        for maker in a.log_id:
            if maker not in maker_list:
                a.log_id.remove(maker)
                print('��{}�� id��¼û��ί������ί��*******************************'.format(maker))
            else:
                print('��{}�� id��¼�Ѿ�ί�в�ί��*******************************'.format(maker))
    else:
        print('���׼��û��id��¼����*******************************')


def run_order_trader_func(c, item=None):
    '''
    �µ����ɽ��������µ�
    '''
    trader_log = get_order(c, c.account, c.account_type)
    # ���ɽ�����
    not_list = [49, 50, 51, 52]
    if trader_log.shape[0] > 0:
        trader_log['���ɽ�'] = trader_log['ί��״̬'].apply(lambda x: '��' if x in not_list else '����')
        trader_log = trader_log[trader_log['���ɽ�'] == '��']
    for item in data:
        name_list = item['�������']
        password_list = item['�����Ȩ��']
        sell_price_code = item['�����۸����']
        buy_price_code = item['����۸����']
        if trader_log.shape[0] > 0:
            trader_log['֤ȯ����'] = trader_log['֤ȯ����'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
            trader_log['�����Ȩ��'] = trader_log['Ͷ�ʱ�ע'].apply(lambda x: str(x).split(',')[0])
            trader_log['����ID'] = trader_log['Ͷ�ʱ�ע'].apply(lambda x: str(x).split(',')[-1])
            trader_log['����ID'] = trader_log['����ID'].astype(str)
            for name, password in zip(name_list, password_list):
                trader_log_new = trader_log[trader_log['�����Ȩ��'] == password]
                if trader_log_new.shape[0] > 0:
                    for stock, amount, trader_type, maker, oder_id in zip(trader_log_new['֤ȯ����'].tolist(),
                                                                          trader_log_new['δ�ɽ�����'].tolist(),
                                                                          trader_log_new['��������'].tolist(),
                                                                          trader_log_new['Ͷ�ʱ�ע'].tolist(),
                                                                          trader_log_new['�������'].tolist()):
                        price = get_price(c, stock)
                        # δ�ɽ�����
                        print(
                            '֤ȯ���룺��{}�� δ�ɽ�������{}���������͡�{}�� Ͷ�ʱ�ע��{}�� ����id��{}��'.format(stock, amount, trader_type,
                                                                                            maker, oder_id))
                        if trader_type == 49:
                            # ����������
                            cancel(oder_id, c.account, c.account_type, c)
                            passorder(c.sell_code, 1101, c.account, str(stock), sell_price_code, 0, int(amount),
                                      str(maker), 1, str(maker), c)
                            print('��ϡ�{}�� ��������������ġ�{}�� ������{}�� �۸�{}��'.format(name, stock, amount, price))
                        elif trader_type == 48:
                            # ����������
                            cancel(oder_id, c.account, c.account_type, c)
                            passorder(c.buy_code, 1101, c.account, str(stock), buy_price_code, 0, int(amount),
                                      str(maker),
                                      1, str(maker), c)
                            print('\n��ϡ�{}�� �������������ġ�{}�� ������{}�� �۸�{}��'.format(name, stock, amount, price))
                        else:
                            print('\n��ϡ�{}�� �������½���δ֪�Ľ�������'.format(name))
                else:
                    print('\n���������µ���ϡ�{}��û��ί������'.format(name))
        else:
            print('\n�����������µ�û��ί������')


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
    print('�ֹ�������{}��'.format(len(positions)))
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
    print('ί��������{}��'.format(len(orders)))
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
            df['ί������'] = [o.m_nVolumeTotalOriginal]
            df['�ɽ�����'] = [o.m_nVolumeTraded]
            df['�������'] = [o.m_strOrderSysID]
            df['δ�ɽ�����'] = df['ί������'] - df['�ɽ�����']
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
    print('�ɽ�������{}��'.format(len(deals)))
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
