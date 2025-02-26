# encoding:gbk
'''
�����������ֶ�����
ͬ��˳����etf���У��Զ�ȫ�г��㷨ѡ�ɽ���
�޸��������
"�˻�":"",
'''
import pandas as pd
import numpy as np
import talib
import time
import os
import json
import requests
from datetime import datetime

text = {
    "�Զ��彻��Ʒ�ֽ���": "�Զ��彻�����ͱ����Ʊ����תծ��etf***********",
    "�˻�": "",
    "�˻�����": "STOCK",
    "�Ƿ�������": "��",
    "����ģʽ˵��": "���/����",
    "����ģʽ": "���",
    "�̶����׽��": 10000,
    "�̶���������": 100,
    "���⽻�ױ������": "���⽻�ױ������",
    "���⽻�ױ��": ['511360.SH', '159651.SZ', '511580.SH', '511380.SH', '159649', '511270.SH',
                     '511030.SH', '511100.SH', '159816.SZ', '159651.SZ', '159972.SZ', '159651.SZ', '511260.SH',
                     '511010.SH', '511220.SH', '511020.SH', '511520.SH', '511060.SH', '511180.SH', '511130.SH',
                     '511090.SH'],
    "���⽻�ױ�Ĺ̶����׽��": 15000,
    "���⽻�ױ�Ĺ̶���������": 100,
    "��������": "��������",
    "�Ƿ���������������": "��",
    "�۸�վ��N��������": 5,
    "�Ƿ��Զ��彻��Ʒ�ֵ���N�վ�������": "��",
    "�Զ��彻��Ʒ�ֵ���N�վ�������": 5,
    "�Զ��彻��Ʒ�ֳ��з���": 50,
    "������ͷ�": 50,
    "����ǰN": 10,
    "��������": 10,
    "�ֹ�����": 10,
    "���뽻���ǵ�������": "�����ǵ�������",
    "����ǵ���": 6,
    "��С�ǵ���": -3,
    "����������": "�Զ�������������*****************",
    "�Ƿ���������˵��": "���߿��Բ��������������ƿ��Կ���",
    "�Ƿ���������": "��",
    "����������": 2,
    "���������": 1,
    "�Ƿ���������": "��",
    "����������": 4,
    "���������": 3,
    "�Ƿ�����������": "��",
    "����": 3,
    "�Ƿ����������": "��",
    "���": -1.5,
    "ʱ������": "ʱ������********",
    "����ʱ���": 4,
    "���׿�ʼʱ��": 9,
    "���׽���ʱ��": 14,
    "�Ƿ�μӼ��Ͼ���": "��",
    "��ʼ���׷���": 0,
    "������������": "������������",
    "����ʱ��": 10,
    "��������": 2,
    "��������": -40,
    '�Զ����Ʊ��': "�Զ����Ʊ������",
    "������": "http://124.220.32.224",
    "�˿�": "8888",
    "��Ȩ��": "xg123456",
}
'''
�Զ����Ʊ�ظ�ʽ
֤ȯ����       ����
513100.SH     ��˹���ETF
159502.SZ     ��������ETF
�ر�ע��֤ȯ����Ҫ���г�.SZ,.SH
'''


class A():
    pass


a = A()


class xg_data:
    '''
    С������api��֧��qmt,����
    '''

    def __init__(self, url='http://124.220.32.224', port=8888, password='123456'):
        '''
        С������api��֧��qmt,����
        url��������ҳ
        port�˿�
        password��Ȩ��
        '''
        self.url = url
        self.port = port
        self.password = password

    def get_user_info(self):
        '''
        ��ȡ�û���Ϣ
        '''
        url = '{}:{}/_dash-update-component'.format(self.url, self.port)
        headers = {'Content-Type': 'application/json'}
        data = {
            "output": "finace_data_table_1.data@e60ed22f488acd1653d4a92a187c4775d06cc39e4afa58da3bee9c8261dcc6a0",
            "outputs": {"id": "finace_data_table_1",
                        "property": "data@e60ed22f488acd1653d4a92a187c4775d06cc39e4afa58da3bee9c8261dcc6a0"},
            "inputs": [{"id": "finace_data_password", "property": "value", "value": self.password},
                       {"id": "finace_data_data_type", "property": "value", "value": "����"},
                       {"id": "finace_data_text", "property": "value",
                        "value": "from trader_tool.stock_data import stock_data\nstock_data=stock_data()\ndf=stock_data.get_stock_hist_data_em(stock='600031',start_date='20210101',end_date='20600101',data_type='D',count=8000)\ndf.to_csv(r'{}\\����\\{}����.csv')\n                \n                "},
                       {"id": "finace_data_run", "property": "value", "value": "����"},
                       {"id": "finace_data_down_data", "property": "value", "value": "����������"}],
            "changedPropIds": ["finace_data_run.value"], "parsedChangedPropsIds": ["finace_data_run.value"]}

        res = requests.post(url=url, data=json.dumps(data), headers=headers)
        text = res.json()
        df = pd.DataFrame(text['response']['finace_data_table_1']['data'])
        return df

    def get_user_def_data(self, func=''):
        '''
        �Զ������ݻ�ȡ
        �������ݿ�
        '''
        text = self.params_func(text=func)
        func = text
        info = self.get_user_info()
        print(info)
        url = '{}:{}/_dash-update-component'.format(self.url, self.port)
        headers = {'Content-Type': 'application/json'}
        data = {
            "output": "finace_data_table.data@e60ed22f488acd1653d4a92a187c4775d06cc39e4afa58da3bee9c8261dcc6a0",
            "outputs": {"id": "finace_data_table",
                        "property": "data@e60ed22f488acd1653d4a92a187c4775d06cc39e4afa58da3bee9c8261dcc6a0"},
            "inputs": [{"id": "finace_data_password", "property": "value", "value": self.password},
                       {"id": "finace_data_data_type", "property": "value", "value": "����"},
                       {"id": "finace_data_text", "property": "value", "value": func},
                       {"id": "finace_data_run", "property": "value", "value": "����"},
                       {"id": "finace_data_down_data", "property": "value", "value": "����������"}],
            "changedPropIds": ["finace_data_run.value"], "parsedChangedPropsIds": ["finace_data_run.value"]}
        res = requests.post(url=url, data=json.dumps(data), headers=headers)
        text = res.json()
        df = pd.DataFrame(text['response']['finace_data_table']['data'])
        return info, df

    def params_func(self, text=''):
        '''
        ��������
        '''
        data_list = []
        f = text.split('\n')
        for i in f:
            text = i.strip().lstrip()
            data_list.append(text)
        func = '\n'.join(data_list)
        return func


def init(c):
    # �˻�
    c.account = text['�˻�']
    # �˻�����
    c.account_type = text['�˻�����']
    # ���׹�Ʊ��
    hold_limit = text['��������']
    # ��ȡ���׹�Ʊ��
    c.url = text['������']
    c.port = text['�˿�']
    c.user_password = text['��Ȩ��']
    print('С���������������֧��')
    print('������{}'.format(c.url))
    print('�˿�{}'.format(c.port))
    print('��Ȩ��{}'.format(c.user_password))
    c.data = xg_data(url=c.url, port=c.port, password=c.user_password)
    # a.trader_df=read_trader_stock(c)
    c.run_time("run_tarder_func", "1nDay", "2024-07-25 09:55:00")
    c.run_time("run_tarder_func", "1nDay", "2024-07-25 14:45:00")
    c.run_time("reverse_repurchase_of_treasury_bonds_1", "1nDay", "2024-07-25 14:57:00")
    c.run_time("run_get_mi_pulse_trader", "2nSecond", "2024-07-25 13:20:00")
    print(get_account(c, c.account, c.account_type))
    print(get_position(c, c.account, c.account_type))
    # passorder(23, 1101, c.account, '513100.SH', 5, 0, 100, '',1,'',c)
    run_tarder_func(c)


# print(run_tarder_func(c))
def handlebar(c):
    # run_tarder_func(c)
    pass


def read_trader_stock(c):
    '''
    ��ȡ���׹�Ʊ��
    '''
    func = '''
		from trader_tool.ths_rq import ths_rq
		rq=ths_rq()
		df=rq.get_etf_hot_rank()
		print(df)
		'''
    info, df = c.data.get_user_def_data(func=func)
    print(df)
    stats = df['����״̬'].tolist()[-1]
    if stats == True:
        df['֤ȯ����'] = df['����']
        df['֤ȯ����'] = df['֤ȯ����'].apply(lambda x: str(x) + '.SH' if str(x)[:2] == '51' else str(x) + '.SZ')
        df['��������'] = df['����']
        df['���1'] = df['��������'].apply(lambda x: str(x)[:2])
        df['���2'] = df['��������'].apply(lambda x: str(x).split('ETF')[0][-2:])
        df = df.drop_duplicates(subset=['���1'])
        df = df.drop_duplicates(subset=['���2'])
        print(df)
    else:
        df = df
    return df


def read_tdx_trader_stock(c, path=r'C:\new_tdx\T0002\blocknew\BUY.blk'):
    '''
    ��ȡͨ���Ű����ѡ�ɽ���
    '''
    try:
        stock_list = []
        with open(r'{}'.format(path), 'r+') as f:
            com = f.readlines()
        for i in com:
            i = i.strip()
            if len(str(i)) > 0:
                stock_list.append(i)
        df = pd.DataFrame()
        df['֤ȯ����'] = stock_list
        df['֤ȯ����'] = df['֤ȯ����'].apply(
            lambda x: str(x)[-6:] + '.SH' if str(x)[0] == '1' else str(x)[-6:] + '.SZ')
        return df
    except:
        print('·��������{}'.format(path))
        df = pd.DataFrame()
        return df


def get_mean_line_analyis(c):
    '''
    ���׾��߷���
    '''
    df = read_trader_stock(c)
    line = text['�۸�վ��N��������']

    hold_stock = get_position(c, c.account, c.account_type)
    if hold_stock.shape[0] > 0:
        hold_stock = hold_stock[hold_stock['�ֲ���'] >= 10]
        if hold_stock.shape[0] > 0:
            hold_stock_list = hold_stock['֤ȯ����'].tolist()
        else:
            hold_stock_list = []
    else:
        hold_stock_list = []

    df['�ֹɼ��'] = df['֤ȯ����'].apply(lambda x: '��' if x in hold_stock_list else '����')
    df = df[df['�ֹɼ��'] == '����']
    print(df)
    select_list = []
    if df.shape[0] > 0:
        for stock in df['֤ȯ����'].tolist():

            try:
                hist = c.get_market_data_ex([], stock_code=[stock], period="1d", count=-1,
                                            start_time='20210101',
                                            end_time='20500101',
                                            dividend_type='front')
                hist = hist[stock]
                stats = tarder_up_mean_line(c, close_list=hist['close'].tolist(), line=line)
                if stats == True:
                    select_list.append('��')
                else:
                    select_list.append('����')
            except:
                print('���׾��߷���{} ����������'.format(stock))
                select_list.append('����')
        df['���׾��߷���'] = select_list
        df = df[df['���׾��߷���'] == '��']
        return df
    else:
        return df


def get_score_analysis(c):
    '''
    ��������
    '''
    print('��������*************')
    min_score = text['������ͷ�']
    df = get_mean_line_analyis(c)
    score_list = []
    if df.shape[0] > 0:
        for stock in df['֤ȯ����'].tolist():
            try:
                hist = c.get_market_data_ex([], stock_code=[stock], period="1d", count=-1,
                                            start_time='20210101',
                                            end_time='20500101',
                                            dividend_type='front')
                hist = hist[stock]
                score = mean_line_models(c, close_list=hist['close'].tolist())
                score_list.append(score)
            except:
                print('{} ��������������'.format(stock))
                score_list.append(0)
        df['����'] = score_list
        df = df[df['����'] >= min_score]
        return df
    else:
        return df


def get_limit_analysis(c):
    '''
    �ǵ�������
    '''
    print('�ǵ�������****************')
    df = get_score_analysis(c)
    min_limit = text['��С�ǵ���']
    max_limit = text['����ǵ���']
    limit_list = []
    if df.shape[0] > 0:
        for stock in df['֤ȯ����'].tolist():
            limit = cacal_limit(c, stock)
            limit_list.append(limit)
        df['�ǵ���'] = limit_list
        df = df[df['�ǵ���'] >= min_limit]
        df = df[df['�ǵ���'] <= max_limit]
        return df
    else:
        return df


def cacal_cycle_analysis(c):
    '''
    ����������
    '''
    print('����������************')
    min_ratio = text['����������']
    is_open = text['�Ƿ���������']
    df = get_limit_analysis(c)
    if is_open == '��':
        print('����������&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        if df.shape[0] > 0:
            cycle_list = []
            df['֤ȯ����'] = df['֤ȯ����'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
            for stock in df['֤ȯ����'].tolist():

                try:
                    hist = hist = c.get_market_data_ex([], stock_code=[stock], period="1w",
                                                       count=-1,
                                                       start_time='20210101',
                                                       end_time='20500101',
                                                       dividend_type='front')
                    hist = hist[stock]
                    signal, markers = six_pulse_excalibur(c, df=hist)
                    cycle_list.append(signal)
                except Exception as e:
                    print(e)
                    cycle_list.append(None)
            df['������'] = cycle_list
            df = df[df['������'] >= min_ratio]
            return df
        else:
            df = pd.DataFrame()
            return df
    else:
        print('������������&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        return df


def cacal_diurnal_cycle(c):
    '''
    ����������
    '''
    print("����������*******************")
    min_ratio = text['����������']
    is_open = text['�Ƿ���������']
    df = cacal_cycle_analysis(c)
    if is_open == '��':
        print('����������&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        if df.shape[0] > 0:
            cycle_list = []
            df['֤ȯ����'] = df['֤ȯ����'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
            for stock in df['֤ȯ����'].tolist():
                try:
                    hist = hist = c.get_market_data_ex([], stock_code=[stock], period="1d",
                                                       count=-1,
                                                       start_time='20210101',
                                                       end_time='20500101',
                                                       dividend_type='front')
                    hist = hist[stock]
                    signal, markers = six_pulse_excalibur(c, df=hist)
                    cycle_list.append(signal)
                except Exception as e:
                    print(e)
                    cycle_list.append(None)
            df['������'] = cycle_list
            print('������************')
            print(df)
            df = df[df['������'] >= min_ratio]

            return df
        else:
            df = pd.DataFrame()
            return df
    else:
        return df


def get_sell_stock_data(c):
    '''
    ��ȡ������Ʊ����
    '''
    print('��ȡ������Ʊ����***********')
    is_del = text['�Ƿ�������']
    is_open_down_mean_line = text['�Ƿ��Զ��彻��Ʒ�ֵ���N�վ�������']
    men_line_n = text['�Զ��彻��Ʒ�ֵ���N�վ�������']
    is_open_week_n_sell = text['�Ƿ���������']
    week_n_sell = text['���������']
    daily_n_sell = text['���������']
    zdf_sell = text['����']
    is_open_zdf_not_sell = text['�Ƿ����������']
    zdf_not_sell = text['���']
    is_open_max_zdf_sell = text['�Ƿ�����������']
    hold_score = text['�Զ��彻��Ʒ�ֳ��з���']
    df = get_position(c, c.account, c.account_type)
    print("�������***************************")
    trader_df = read_trader_stock(c)
    if trader_df.shape[0] > 0:
        trader_df['֤ȯ����'] = trader_df['֤ȯ����'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
        trader_stock_list = trader_df['֤ȯ����'].tolist()

    else:
        trader_stock_list = []
    if is_del == '��':
        print('�������**************')
        if df.shape[0] > 0:
            df['֤ȯ����'] = df['֤ȯ����'].astype(str)
            df['�������'] = df['֤ȯ����'].apply(lambda x: '��' if x in trader_stock_list else '����')
            df1 = df[df['�������'] == '����']
            df = df[df['�������'] == '��']
            print('�Զ����Ʊ�ر��************************')
            print(df)
            print('�����Զ��彻�׹�Ʊ�ı��*******************')
            print(df1)
        else:
            df = df
    if df.shape[0] > 0:
        week_cycle_list = []
        for stock in df['֤ȯ����'].tolist():
            try:
                hist = c.get_market_data_ex([], stock_code=[stock], period="1w", count=-1,
                                            start_time='20210101',
                                            end_time='20500101',
                                            dividend_type='front')
                hist = hist[stock]
                signal, markers = six_pulse_excalibur(c, df=hist)
                week_cycle_list.append(signal)
            except Exception as e:
                print(e)
                week_cycle_list.append(None)
        df['������'] = week_cycle_list
        daily_cycle_list = []
        for stock in df['֤ȯ����'].tolist():
            try:
                hist = c.get_market_data_ex([], stock_code=[stock], period="1d", count=-1,
                                            start_time='20210101',
                                            end_time='20500101',
                                            dividend_type='front')
                hist = hist[stock]
                signal, markers = six_pulse_excalibur(c, df=hist)
                daily_cycle_list.append(signal)
            except Exception as e:
                print(e)
                daily_cycle_list.append(None)
        df['������'] = daily_cycle_list
        if is_open_down_mean_line == '��':
            print('�������߷���********')
            down_list = []
            for stock in df['֤ȯ����'].tolist():
                try:
                    hist = c.get_market_data_ex([], stock_code=[stock], period="1d", count=-1,
                                                start_time='20210101',
                                                end_time='20500101',
                                                dividend_type='front')
                    hist = hist[stock]
                    down = tarder_down_mean_line(c, close_list=hist['close'].tolist(), line=men_line_n)
                    down_list.append(down)
                except Exception as e:
                    print(e)
                    down_list.append('��')
            df['���߷���'] = down_list
        else:
            df['���߷���'] = '����'
        if is_open_max_zdf_sell == '��':
            zdf_list = []
            for stock in df['֤ȯ����'].tolist():
                try:
                    zdf = cacal_limit(c, stock)
                    zdf_list.append(zdf)
                except Exception as e:
                    print(e)
                    zdf_list.append(-40)
            df['�ǵ���'] = zdf_list
        else:
            df['�ǵ���'] = -40
        mean_score_list = []
        for stock in df['֤ȯ����'].tolist():
            try:
                hist = hist = c.get_market_data_ex([], stock_code=[stock], period="1d",
                                                   count=-1,
                                                   start_time='20210101',
                                                   end_time='20500101',
                                                   dividend_type='front')
                hist = hist[stock]
                score = mean_line_models(c, close_list=hist['close'].tolist())
                mean_score_list.append(score)
            except Exception as e:
                print('�ֹɾ��߼���������{}', format(stock))
                mean_score_list.append(100)
        df['���Ƶ÷�'] = mean_score_list
        sell_stock_list = []
        print('(((((((((((((((�ֹɷ���((((((((((((((((9')
        print(df)
        for stock, week_cycle, daily_cycle, score, down, zdf in zip(df['֤ȯ����'],
                                                                    df['������'], df['������'], df['���Ƶ÷�'],
                                                                    df['���߷���'], df['�ǵ���']):
            if is_open_week_n_sell == '��' and week_cycle <= week_n_sell:
                print('{} ������{} С��ƽ��������{} ƽ��'.format(stock, week_cycle, week_cycle))
                sell_stock_list.append(stock)
            elif daily_cycle <= daily_n_sell:
                print('{} ������{} С��ƽ��������{} ƽ��'.format(stock, daily_cycle, daily_n_sell))
                sell_stock_list.append(stock)
            elif score < hold_score:
                print('{} ����{} С�ڳ��з���{} ƽ��'.format(stock, score, hold_score))
                sell_stock_list.append(stock)
            elif down == True:
                print('{} ���ƾ���{} ƽ��'.format(stock, men_line_n))
                sell_stock_list.append(stock)
            elif zdf >= zdf_sell:
                print('{} �ǵ���{} ����ƽ���ǵ���{} ƽ��'.format(stock, zdf, zdf_sell))
                sell_stock_list.append(stock)
            else:
                print('{} ����������ģ�ͼ�������'.format(stock))
            sell_stock_list = list(set(sell_stock_list))
            for stock, zdf in zip(df['֤ȯ����'], df['�ǵ���']):
                if zdf <= zdf_not_sell:
                    if is_open_zdf_not_sell == '��':
                        try:
                            sell_stock_list.remove(stock)
                            print('{}����������� �ǵ���{} С�ڴ���ǵ���'.format(stock, zdf, zdf_not_sell))
                        except:
                            pass
                    else:
                        pass
                else:
                    pass
        df['���óֹ�'] = df['֤ȯ����'].apply(lambda x: '��' if x in sell_stock_list else '����')
        df = df[df['���óֹ�'] == '��']
        return df
    else:
        sell_df = pd.DataFrame()
        return sell_df


def get_buy_sell_stock_data(c):
    '''
    ��ȡ��������
    '''
    print('��ȡ��������*********')
    hold_limit = text['�ֹ�����']
    hold_stock = get_position(c, c.account, c.account_type)
    if hold_stock.shape[0] > 0:
        print(hold_stock, '************')
        hold_stock = hold_stock[hold_stock['�ֲ���'] >= 10]
        if hold_stock.shape[0] > 0:
            hold_stock_list = hold_stock['֤ȯ����'].tolist()
            hold_amount = hold_stock.shape[0]
        else:
            hold_stock_list = []
            hold_amount = 0
    else:
        hold_stock_list = []
        hold_amount = 0
    buy_df = cacal_diurnal_cycle(c)
    print('���׹�Ʊ��*************')
    buy_df['����״̬'] = 'δ��'
    print(buy_df)
    if buy_df.shape[0] > 0:
        def select_data(stock):
            if str(stock) in hold_stock_list:
                return '�ֹɳ�������'
            else:
                return 'û�гֹ�'

        buy_df['�ֹɼ��'] = buy_df['֤ȯ����'].apply(select_data)
        buy_df = buy_df[buy_df['�ֹɼ��'] == 'û�гֹ�']
    sell_df = get_sell_stock_data(c)
    sell_df['����״̬'] = 'δ��'
    if sell_df.shape[0] > 0:
        sell_df['֤ȯ����'] = sell_df['֤ȯ����'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
        sell_stock_list = sell_df['֤ȯ����'].tolist()
        sell_amount = len(sell_stock_list)
    else:
        sell_amount = 0
    print('������Ʊ**********************')
    print(sell_df)
    av_buy = (hold_limit - hold_amount) + sell_amount
    if av_buy >= hold_limit:
        av_buy = hold_limit
    else:
        av_buy = av_buy
    buy_df = buy_df[:av_buy]
    print('����ı��***************************')
    print(buy_df)
    return buy_df, sell_df


def run_tarder_func(c):
    '''
    ���н��׺���
    '''
    trader_models = text['����ģʽ']
    fix_value = text['�̶����׽��']
    fix_amount = text['�̶����׽��']
    sep_fix_value = text['���⽻�ױ�Ĺ̶����׽��']
    sep_fix_amount = text['���⽻�ױ�Ĺ̶���������']
    sep_stock_list = text['���⽻�ױ��']
    if check_is_trader_date_1():
        # ����������
        buy_df, sell_df = get_buy_sell_stock_data(c)
        if sell_df.shape[0] > 0:
            for stock, hold_amount, av_amount in zip(sell_df['֤ȯ����'], sell_df['�ֲ���'], sell_df['��������']):
                try:
                    if av_amount >= 10:
                        print(
                            '{} ��������{} ��������{}����0 ��������{}'.format(stock, hold_amount, av_amount, av_amount))
                        passorder(24, 1101, c.account, stock, 5, 0, av_amount, '', 1, '', c)
                    else:
                        print('{} ��������{} ��������{}����0 ��������{} ������'.format(stock, hold_amount, av_amount,
                                                                                       av_amount))
                except:
                    print('{}����������'.format(stock))
        else:
            print('û������������')
        # ����
        if buy_df.shape[0] > 0:
            for stock in buy_df['֤ȯ����'].tolist():
                if stock in sep_stock_list:
                    print('{}������������*********'.format(stock))
                    fix_value = sep_fix_value
                    volume = sep_fix_amount
                else:
                    fix_value = text['�̶����׽��']
                    volume = fix_amount
                print(stock, fix_value)
                if trader_models == '���':
                    print('{}����ģʽ*******'.format(stock))
                    tader_type, amount, price = order_stock_value(c, c.account, c.account_type, stock, fix_value, 'buy')
                    print(tader_type, amount, price)
                    if tader_type == 'buy' and amount >= 10:
                        passorder(23, 1101, c.account, str(stock), 5, 0, amount, '', 1, '', c)
                        # passorder(23, 1101, c.account, str('513100.SH'), 5, 0, 100, '',1,'',c)
                        print('{} ���¼۸� ����{} Ԫ'.format(stock, fix_value))
                    else:
                        print('{}����ģʽ���벻��*******'.format(stock))
                else:
                    print('{}��������ģʽ*******'.format(stock))
                    passorder(23, 1101, c.account, str(stock), 5, 0, volume, '', 1, '', c)
                    print('{} ���¼۸� ����{} ����'.format(stock, volume))

        else:
            print('û����������')
    else:
        print('{} Ŀǰ���ٽ���ʱ��'.format(datetime.now()))


def run_get_mi_pulse_trader(c):
    '''
    ��������ģ��
    '''
    sell_limit = text['��������']
    buy_limit = text['��������']
    n = text['����ʱ��']
    fix_value = text['�̶����׽��']
    hold_stock = get_position(c, c.account, c.account_type)
    # hold_stock['֤ȯ����']='600031.SH'
    if check_is_trader_date_1():
        if hold_stock.shape[0] > 0:
            hold_stock = hold_stock[hold_stock['�ֲ���'] >= 10]
            if hold_stock.shape[0] > 0:
                for stock, hold_amount, av_amount in zip(hold_stock['֤ȯ����'], hold_stock['�ֲ���'],
                                                         hold_stock['��������']):
                    try:
                        stats = get_mi_pulse_trader(c, stock=stock, n=n, x1=sell_limit, x2=buy_limit)
                        if stats == 'sell':
                            if av_amount >= 10:
                                print('{} ��������{} ��������{}����0 ��������{}'.format(stock, hold_amount, av_amount,
                                                                                        av_amount))
                                passorder(24, 1101, c.account, stock, 5, 0, av_amount, '', 1, '', c)
                            else:
                                print('{} ��������{} ��������{}����0 ��������{} ������'.format(stock, hold_amount,
                                                                                               av_amount, av_amount))
                        elif stats == 'buy':
                            print('{} ���¼۸� ����{} Ԫ'.format(stock, fix_value))
                            passorder(23, 1102, c.account, stock, 5, -1, fix_value, c)
                        else:
                            print('ʱ��{} ��Ʊ{} �����������������'.format(datetime.now(), stock))
                    except:
                        print('{} {}����������������ܲ��ǽ�����'.format(stock, datetime.now()))
            else:
                print('{}��������û�гֹ�'.format(datetime.now()))
        else:
            print('{}��������û�гֹ�'.format(datetime.now()))
    else:
        print('ʱ��{} �������岻�ǽ���ʱ��'.format(datetime.now()))


def get_mi_pulse_trader(c, stock='603496.SH', n=10, x1=2, x2=-40):
    '''
    �����������
    '''
    start_time = ''.join(str(datetime.now())[:10].split('-'))
    c.subscribe_quote(
        stock_code=stock,
        period='follow',
        dividend_type='follow', )
    data = c.get_market_data_ex(stock_code=[stock], period='tick',
                                start_time=start_time, end_time='20500101', count=-1, subscribe=True)
    df = data[stock]
    df = df[-25 * n:]
    df['return'] = (df['lastPrice'].pct_change() * 100).cumsum()
    zdf_list = df['return'].tolist()
    zdf_1 = zdf_list[-1]
    zdf = zdf_list[-1] - zdf_list[1]
    print('��Ʊ{} {}���� �ǵ���{}'.format(stock, n, zdf))
    if zdf >= x1:
        print('ʱ��{} ����{} ���� {}���ӵ����� {} Ŀǰ�ǵ���{},�����ǵ���{}'.format(datetime.now(), stock, n, x1, zdf_1,
                                                                                    zdf))
        return 'sell'
    elif zdf <= x2:
        print('ʱ��{} ����{} ���� {}���ӵ����� {} Ŀǰ�ǵ���{} �����ǵ���{}'.format(datetime.now(), stock, n, x1, zdf_1,
                                                                                    zdf))
        return 'buy'
    else:
        print('ʱ��{} {} û�д��� {}���ӵ����� Ŀǰ�ǵ���{} �����ǵ���{} '.format(datetime.now(), stock, n, zdf_1, zdf))
        return ''


def reverse_repurchase_of_treasury_bonds_1(c, buy_ratio=1):
    '''
    ��ծ��ع�1,�µĺ���
    �������buy_ratio
    '''
    # �Խ��׻ص����ж��ģ����ĺ�����յ��������ƣ�����0��ʾ���ĳɹ�
    account = get_account(c, c.account, c.account_type)
    print(account)
    av_cash = account['���ý��']
    av_cash = float(av_cash)
    av_cash = av_cash * buy_ratio
    # stock_code_sh = '204001.SH'
    # ͳһ������
    stock_code_sh = '131810.SZ'
    stock_code_sz = '131810.SZ'
    price_sh = get_price(c, stock_code_sh)
    price_sz = get_price(c, stock_code_sz)
    bidPrice1 = max(price_sh, price_sz)
    if price_sh >= price_sz:
        stock_code = stock_code_sh
    else:
        stock_code = stock_code_sz
    print(stock_code, bidPrice1)
    price = bidPrice1
    stock = stock_code
    # �µ�������Ҫ��1000
    amount = int(av_cash / 1000)
    # ����ȡ��1000�ı���
    amount = math.floor(amount / 10) * 100
    # ���Ǯsell
    print('��ʼ��ع�***********')
    if amount > 0:
        sell(c, stock=stock, amount=amount, price=price)
        text = '��ծ��ع��������� ����{} �۸�{} ����{} �������{}'.format(stock, price, amount, fix_result_order_id)
        return '���׳ɹ�', text
    else:
        text = '��ծ��ع����� ���{} �۸�{} ί������{}С��0������'.format(stock, price, amount)
        print('�˻�û�п��Ե�Ǯ@@@@@@@@@@@@@@@@@@@')
        return '����ʧ��', text


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


def cacal_limit(c, stock):
    '''
    �����ǵ���
    '''
    try:
        tick = c.get_full_tick(stock_code=[stock])
        tick = tick[stock]
        limit = ((tick['lastPrice'] - tick['lastClose']) / tick['lastClose']) * 100
        return limit
    except:
        print('{}�ǵ�������������'.format(stock))
        return -40


def get_price(c, stock):
    '''
    ��ȡ���¼۸�
    '''
    tick = c.get_full_tick(stock_code=[stock])
    tick = tick[stock]
    price = tick['lastPrice']
    return price


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


def mean_line_models(c, close_list=[], x1=3, x2=5, x3=10, x4=15, x5=20):
    '''
    ����ģ��
    ����ģ��
    5��10��20��30��60
    '''
    df = pd.DataFrame()
    df['close'] = close_list
    # df=self.bond_cov_data.get_cov_bond_hist_data(stock=stock,start=start_date,end=end_date,limit=1000000000)
    df1 = pd.DataFrame()
    df1['x1'] = df['close'].rolling(window=x1).mean()
    df1['x2'] = df['close'].rolling(window=x2).mean()
    df1['x3'] = df['close'].rolling(window=x3).mean()
    df1['x4'] = df['close'].rolling(window=x4).mean()
    df1['x5'] = df['close'].rolling(window=x5).mean()
    score = 0
    # �ӷֵ����
    mean_x1 = df1['x1'].tolist()[-1]
    mean_x2 = df1['x2'].tolist()[-1]
    mean_x3 = df1['x3'].tolist()[-1]
    mean_x4 = df1['x4'].tolist()[-1]
    mean_x5 = df1['x5'].tolist()[-1]
    # ����2�����߽��бȽ�
    if mean_x1 >= mean_x2:
        score += 25
    if mean_x2 >= mean_x3:
        score += 25
    if mean_x3 >= mean_x4:
        score += 25
    if mean_x4 >= mean_x5:
        score += 25
    return score


def tarder_up_mean_line(c, close_list=[], line=5):
    '''
    վ�Ͻ��׾���
    '''
    price = close_list[-1]
    df = pd.DataFrame()
    df['close'] = close_list
    df['line'] = df['close'].rolling(line).mean()
    mean_line = df['line'].tolist()[-1]
    if price >= mean_line:
        return True
    else:
        return False


def tarder_down_mean_line(c, close_list=[], line=5):
    '''
    ���ƽ��׾���
    '''
    price = close_list[-1]
    df = pd.DataFrame()
    df['close'] = close_list
    df['line'] = df['close'].rolling(line).mean()
    mean_line = df['line'].tolist()[-1]
    if price < mean_line:
        return True
    else:
        return False


def stock_stop_up(c, close_list=[], limit=3):
    '''
    ����ֹӯ
    '''
    df = pd.DataFrame()
    df['close'] = close_list
    df['day_limit'] = df['close'].pct_change() * 100
    day_limit = df['day_limit'].tolist()[-1]
    if limit >= day_limit:
        return True
    else:
        return False


def stock_stop_up(c, close_list=[], limit=-2):
    '''
    ����ֹ��
    '''
    df = pd.DataFrame()
    df['close'] = close_list
    df['day_limit'] = df['close'].pct_change() * 100
    day_limit = df['day_limit'].tolist()[-1]
    if day_limit <= limit:
        return True
    else:
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
            data = pd.concat([data, df], ignore_index=True)
    else:
        data = pd.DataFrame()


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


def order_target_amount(c, accountid, datatype, stock, price, target_amount, com_ratio=0.0001):
    '''
    Ŀ�꽻������
    '''
    account = get_account(c, accountid, datatype)
    # ����ʹ�õ��ֽ�
    av_cash = account['���ý��']
    position = get_position(c, accountid, datatype)
    if position.shape[0] > 0:
        position[position['�ֲ���'] >= 10]
        if position.shape[0] > 0:
            hold_amount = position['�ֲ���'].tolist()[-1]
            av_amount = position['��������'].tolist()[-1]
        else:
            hold_amount = 0
            av_amount = 0
    else:
        hold_amount = 0
        av_amount = 0
    # ���Խ��׵�����
    av_trader_amount = target_amount - hold_amount
    # ��������ռ�
    if av_trader_amount >= 10:
        # ����ļ�ֵ
        value = av_trader_amount * price
        # ������
        com = value * com_ratio
        if av_cash >= value + com:
            print('{} Ŀ������{} ��������{} ��������{} ��������{} �����ʽ�{} ���������ʽ�{} ����'.format(stock,
                                                                                                         target_amount,
                                                                                                         hold_amount,
                                                                                                         av_amount,
                                                                                                         av_trader_amount,
                                                                                                         av_cash,
                                                                                                         value))
            return 'buy', price, av_trader_amount
        else:
            print('{} Ŀ������{} ��������{} ��������{} ��������{} �����ʽ�{} С�������ʽ�{} ������'.format(stock,
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
            print('{} Ŀ������{} ��������{} ��������{}���� ��������{} ����'.format(stock, target_amount, hold_amount,
                                                                                   av_amount, av_trader_amount))
            return 'sell', price, -av_trader_amount
        else:
            print(
                '{} Ŀ������{} ��������{} ��������{}С�� ��������{} ����ȫ��'.format(stock, target_amount, hold_amount,
                                                                                     av_amount, av_trader_amount))
            return 'sell', price, -av_amount
    else:
        print('{} Ŀ������{} ��������{}һ��������'.format(stock, target_amount, hold_amount))
        return '', '', ''


def six_pulse_excalibur(c, df):
    '''
    ������
    '''
    markers = 0
    signal = 0
    # df=self.data.get_hist_data_em(stock=stock)
    CLOSE = df['close']
    LOW = df['low']
    HIGH = df['high']
    DIFF = EMA(CLOSE, 8) - EMA(CLOSE, 13)
    DEA = EMA(DIFF, 5)
    # �������DIFF>DEA ��1��λ�ñ��1��ͼ��
    # DRAWICON(DIFF>DEA,1,1);
    markers += IF(DIFF > DEA, 1, 0)[-1]
    # �������DIFF<DEA ��1��λ�ñ��2��ͼ��
    # DRAWICON(DIFF<DEA,1,2);
    markers += IF(DIFF < DEA, 1, 0)[-1]
    # DRAWTEXT(ISLASTBAR=1,1,'. MACD'),COLORFFFFFF;{΢�Ź��ں�:���ݷ���������}
    ABC1 = DIFF > DEA
    signal += IF(ABC1, 1, 0)[-1]
    �����г�1 = (CLOSE - LLV(LOW, 8)) / (HHV(HIGH, 8) - LLV(LOW, 8)) * 100
    K = SMA(�����г�1, 3, 1)
    D = SMA(K, 3, 1)
    # �������k>d ��2��λ�ñ��1��ͼ��
    markers += IF(K > D, 1, 0)[-1]
    # DRAWICON(K>D,2,1);
    markers += IF(K < D, 1, 0)[-1]
    # DRAWICON(K<D,2,2);
    # DRAWTEXT(ISLASTBAR=1,2,'. KDJ'),COLORFFFFFF;
    ABC2 = K > D
    signal += IF(ABC2, 1, 0)[-1]
    ָ��Ӫ�� = REF(CLOSE, 1)
    RSI1 = (SMA(MAX(CLOSE - ָ��Ӫ��, 0), 5, 1)) / (SMA(ABS(CLOSE - ָ��Ӫ��), 5, 1)) * 100
    RSI2 = (SMA(MAX(CLOSE - ָ��Ӫ��, 0), 13, 1)) / (SMA(ABS(CLOSE - ָ��Ӫ��), 13, 1)) * 100
    markers += IF(RSI1 > RSI2, 1, 0)[-1]
    # DRAWICON(RSI1>RSI2,3,1);
    markers += IF(RSI1 < RSI2, 1, 0)[-1]
    # DRAWICON(RSI1<RSI2,3,2);
    # DRAWTEXT(ISLASTBAR=1,3,'. RSI'),COLORFFFFFF;
    ABC3 = RSI1 > RSI2
    signal += IF(ABC3, 1, 0)[-1]
    �����г� = -(HHV(HIGH, 13) - CLOSE) / (HHV(HIGH, 13) - LLV(LOW, 13)) * 100
    LWR1 = SMA(�����г�, 3, 1)
    LWR2 = SMA(LWR1, 3, 1)
    # DRAWICON(LWR1>LWR2,4,1);
    markers += IF(LWR1 > LWR2, 1, 0)[-1]
    # DRAWICON(LWR1<LWR2,4,2);
    markers += IF(LWR1 < LWR2, 1, 0)[-1]
    # DRAWTEXT(ISLASTBAR=1,4,'. LWR'),COLORFFFFFF;
    ABC4 = LWR1 > LWR2
    signal += IF(ABC4, 1, 0)[-1]
    BBI = (MA(CLOSE, 3) + MA(CLOSE, 5) + MA(CLOSE, 8) + MA(CLOSE, 13)) / 4
    # DRAWICON(CLOSE>BBI,5,1);
    markers += IF(CLOSE > BBI, 1, 0)[-1]
    # DRAWICON(CLOSE<BBI,5,2);
    markers += IF(CLOSE < BBI, 1, 0)[-1]
    # DRAWTEXT(ISLASTBAR=1,5,'. BBI'),COLORFFFFFF;
    ABC10 = 7
    ABC5 = CLOSE > BBI
    signal += IF(ABC5, 1, 0)[-1]
    MTM = CLOSE - REF(CLOSE, 1)
    MMS = 100 * EMA(EMA(MTM, 5), 3) / EMA(EMA(ABS(MTM), 5), 3)
    MMM = 100 * EMA(EMA(MTM, 13), 8) / EMA(EMA(ABS(MTM), 13), 8)
    markers += IF(MMS > MMM, 1, 0)[-1]
    # DRAWICON(MMS>MMM,6,1);
    markers += IF(MMS < MMM, 1, 0)[-1]
    # DRAWICON(MMS<MMM,6,2);
    # DRAWTEXT(ISLASTBAR=1,6,'. ZLMM'),COLORFFFFFF;
    ABC6 = MMS > MMM
    signal += IF(ABC6, 1, 0)[-1]
    return signal, markers


def RD(N, D=3):
    # ��������ȡ3λС��
    return np.round(N, D)


def RET(S, N=1):
    # �������е�����N��ֵ,Ĭ�Ϸ������һ��
    return np.array(S)[-N]


def ABS(S
        ):
    # ����N�ľ���ֵ
    return np.abs(S)


def MAX(S1, S2):
    # ����max
    return np.maximum(S1, S2)


def MIN(S1, S2):
    # ����min
    return np.minimum(S1, S2)


def IF(S, A, B):
    # ���в����ж� return=A  if S==True  else  B
    return np.where(S, A, B)


def AND(S1, S2):
    # and
    return np.logical_and(S1, S2)


def OR(S1, S2):
    # or
    return np.logical_or(S1, S2)


def RANGE(A, B, C):
    '''
    �ڼ亯��
    B<=A<=C
    '''
    df = pd.DataFrame()
    df['select'] = A.tolist()
    df['select'] = df['select'].apply(lambda x: True if (x >= B and x <= C) else False)
    return df['select']


def REF(S, N=1):  # �������������ƶ�N,��������(shift������NAN)
    return pd.Series(S).shift(N).values


def DIFF(S, N=1):  # ǰһ��ֵ����һ��ֵ,ǰ������nan
    return pd.Series(S).diff(N).values  # np.diff(S)ֱ��ɾ��nan������һ��


def STD(S, N):  # �����е�N�ձ�׼���������
    return pd.Series(S).rolling(N).std(ddof=0).values


def SUM(S, N):  # ��������N���ۼƺͣ���������    N=0�����������������
    return pd.Series(S).rolling(N).sum().values if N > 0 else pd.Series(S).cumsum().values


def CONST(S):  # ��������S����ֵ��ɳ�������
    return np.full(len(S), S[-1])


def HHV(S, N):  # HHV(C, 5) ���5��������߼�
    return pd.Series(S).rolling(N).max().values


def LLV(S, N):  # LLV(C, 5) ���5��������ͼ�
    return pd.Series(S).rolling(N).min().values


def HHVBARS(S, N):  # ��N������S���ֵ����ǰ������, ��������
    return pd.Series(S).rolling(N).apply(lambda x: np.argmax(x[::-1]), raw=True).values


def LLVBARS(S, N):  # ��N������S���ֵ����ǰ������, ��������
    return pd.Series(S).rolling(N).apply(lambda x: np.argmin(x[::-1]), raw=True).values


def MA(S, N):  # �����е�N�ռ��ƶ�ƽ��ֵ����������
    return pd.Series(S).rolling(N).mean().values


def EMA(S, N):  # ָ���ƶ�ƽ��,Ϊ�˾��� S>4*N  EMA������Ҫ120����     alpha=2/(span+1)
    return pd.Series(S).ewm(span=N, adjust=False).mean().values


def SMA(S, N, M=1):  # �й�ʽ��SMA,������Ҫ120���ڲž�ȷ (ѩ��180����)    alpha=1/(1+com)
    return pd.Series(S).ewm(alpha=M / N, adjust=False).mean().values  # com=N-M/M


def DMA(S, A):  # ��S�Ķ�̬�ƶ�ƽ����A��ƽ������,���� 0<A<1  (��Ϊ���ĺ�������ָ�꣩
    return pd.Series(S).ewm(alpha=A, adjust=True).mean().values


def WMA(S, N):  # ͨ����S���е�N�ռ�Ȩ�ƶ�ƽ�� Yn = (1*X1+2*X2+3*X3+...+n*Xn)/(1+2+3+...+Xn)
    return pd.Series(S).rolling(N).apply(lambda x: x[::-1].cumsum().sum() * 2 / N / (N + 1), raw=True).values


def AVEDEV(S, N):  # ƽ������ƫ��  (��������ƽ��ֵ�ľ��Բ��ƽ��ֵ)
    return pd.Series(S).rolling(N).apply(lambda x: (np.abs(x - x.mean())).mean()).values


def SLOPE(S, N):  # ��S����N���ڻ����Իع�б��
    return pd.Series(S).rolling(N).apply(lambda x: np.polyfit(range(N), x, deg=1)[0], raw=True).values


def FORCAST(S, N):  # ����S����N���ڻ����Իع���Ԥ��ֵ�� jqz1226�Ľ������г�
    return pd.Series(S).rolling(N).apply(lambda x: np.polyval(np.polyfit(range(N), x, deg=1), N - 1), raw=True).values


def LAST(S, A, B):  # ��ǰA�յ�ǰB��һֱ����S_BOOL����, Ҫ��A>B & A>0 & B>=0
    return np.array(pd.Series(S).rolling(A + 1).apply(lambda x: np.all(x[::-1][B:]), raw=True), dtype=bool)


# ------------------   1����Ӧ�ò㺯��(ͨ��0�����ĺ���ʵ�֣� ----------------------------------
def COUNT(S, N):  # COUNT(CLOSE>O, N):  ���N������S_BOO������  True������
    return SUM(S, N)


def EVERY(S, N):  # EVERY(CLOSE>O, 5)   ���N���Ƿ���True
    return IF(SUM(S, N) == N, True, False)


def EXIST(S, N):  # EXIST(CLOSE>3010, N=5)  n�����Ƿ����һ�����3000��
    return IF(SUM(S, N) > 0, True, False)


def FILTER(S, N):  # FILTER������S���������󣬽����N�����ڵ�������Ϊ0, FILTER(C==H,5)
    for i in range(len(S)): S[i + 1:i + 1 + N] = 0 if S[i] else S[i + 1:i + 1 + N]
    return S  # ����FILTER(C==H,5) ��ͣ�󣬺�5�첻�ٷ����ź�


def BARSLAST(S):  # ��һ��������������ǰ������, BARSLAST(C/REF(C,1)>=1.1) ��һ����ͣ�����������
    M = np.concatenate(([0], np.where(S, 1, 0)))
    for i in range(1, len(M)):  M[i] = 0 if M[i] else M[i - 1] + 1
    return M[1:]


def BARSLASTCOUNT(S):  # ͳ����������S������������        by jqz1226
    rt = np.zeros(len(S) + 1)  # BARSLASTCOUNT(CLOSE>OPEN)��ʾͳ������������������
    for i in range(len(S)): rt[i + 1] = rt[i] + 1 if S[i] else rt[i + 1]
    return rt[1:]


def BARSSINCEN(S, N):  # N�����ڵ�һ��S�������������ڵ�������,NΪ����  by jqz1226
    return pd.Series(S).rolling(N).apply(lambda x: N - 1 - np.argmax(x) if np.argmax(x) or x[0] else 0,
                                         raw=True).fillna(0).values.astype(int)


def CROSS(S1, S2):  # �ж����Ͻ�洩Խ CROSS(MA(C,5),MA(C,10))  �ж��������洩Խ CROSS(MA(C,10),MA(C,5))
    return np.concatenate(([False], np.logical_not((S1 > S2)[:-1]) & (S1 > S2)[1:]))  # ��ʹ��0������,��ֲ����  by jqz1226


def CROSS_UP(S1, S2):  # �ж����Ͻ�洩Խ CROSS(MA(C,5),MA(C,10))  �ж��������洩Խ CROSS(MA(C,10),MA(C,5))
    return np.concatenate(([False], np.logical_not((S1 > S2)[:-1]) & (S1 > S2)[1:]))  # ��ʹ��0������,��ֲ����  by jqz1226


def CROSS_DOWN(S1, S2):
    return np.concatenate(([False], np.logical_not((S1 < S2)[:-1]) & (S1 < S2)[1:]))  # ��ʹ��0������,��ֲ����  by jqz1226


def LONGCROSS(S1, S2, N):  # ������ά��һ�����ں󽻲�,S1��N�����ڶ�С��S2,�����ڴ�S1�·����ϴ���S2ʱ����1,���򷵻�0
    return np.array(np.logical_and(LAST(S1 < S2, N, 1), (S1 > S2)), dtype=bool)  # N=1ʱ��ͬ��CROSS(S1, S2)


def VALUEWHEN(S, X):  # ��S��������ʱ,ȡX�ĵ�ǰֵ,����ȡVALUEWHEN���ϸ�����ʱ��Xֵ   by jqz1226
    return pd.Series(np.where(S, X, np.nan)).ffill().values