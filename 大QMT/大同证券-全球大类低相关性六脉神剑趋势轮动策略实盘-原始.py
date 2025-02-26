# encoding:gbk
'''
�����������ֶ����ԣ�
�Զ�����Ĺ�Ʊ��
�����Լ����˻��Ϳ���
"�˻�":"111111",
'''
import pandas as pd
import numpy as np
import talib
import time
from datetime import datetime
import math

text = {
    "�Զ��彻��Ʒ�ֽ���": "�Զ��彻�����ͱ����Ʊ����תծ��etf***********",
    "�˻�": "",
    "�˻�����": "STOCK",
    "�Ƿ�������": "��",
    "����ģʽ˵��": "���/����",
    "����ģʽ": "���",
    "�̶����׽��": 1000,
    "�̶���������": 100,
    "���⽻�ױ������": "���⽻�ױ������",
    "���⽻�ױ��": ['511360.SH', '159651.SZ', '511580.SH', '511380.SH', '159649', '511270.SH',
                     '511030.SH', '511100.SH', '159816.SZ', '159651.SZ', '159972.SZ', '159651.SZ', '511260.SH',
                     '511010.SH', '511220.SH',
                     '511020.SH', '511520.SH', '511060.SH', '511180.SH', '511130.SH', '511090.SH'],
    "���⽻�ױ�Ĺ̶����׽��": 15000,
    "���⽻�ױ�Ĺ̶���������": 100,
    "����ǰN": 10,
    "��������": 10,
    "�ֹ�����": 10,
    "������������": "������������******************",
    "�Զ���ѡ�ɺ���": {
        "�ǵ���": "cacal_limit()",
        "�۸�": "get_price()",
        "����": "mean_line_models()",
        "վ�Ͼ���": "tarder_up_mean_line(line=5)",
        "���ƾ���": "tarder_down_mean_line(line=5)",
        "����ֹӯ": "stock_stop_up(limit=3)",
        "����ֹ��": "stock_stop_up(limit=-6)",
        "������": "six_pulse_excalibur()"
    },
    "�Զ���ѡ������ѡ��": {
        "�ǵ���": [0, 3],
        "����": [25, 100],
        "վ�Ͼ���": [1],
        "����ֹӯ": [1],
        "������": [4, 6]
    },
    "������������": "������������*********",
    "�Զ�����������": {
        "����": "mean_line_models()",
        "���ƾ���": "tarder_down_mean_line(line=5)",
        "����ֹ��": "stock_stop_up(limit=-6)",
        "������": "six_pulse_excalibur()"
    },
    "�Զ�����������ѡ��": {
        "����": [0, 25],
        "���ƾ���": [1],
        "����ֹ��": [1],
        "������": [0, 3]
    },

    "ʱ������": "ʱ������********",
    "����ʱ���": 4,
    "���׿�ʼʱ��": 9,
    "���׽���ʱ��": 14,
    "�Ƿ�μӼ��Ͼ���": "��",
    "��ʼ���׷���": 0,
    '�Զ����Ʊ��': "�Զ����Ʊ������",
    "��Ʊ������": "��������10�Ĺ�Ʊ������",
    "��Ʊ��": ['513100.SH', '511130.SH', '159937.SZ', '513350.SH', '512890.SH',
               '159915.SZ', '513500.SH', '159985.SZ', '159981.SZ', '159980.SZ',
               '513300.SH', '159680.SZ', '511090.SH', '513400.SH', '159934.SZ'],
    "��Ʊ������": ['��˹���ETF', '30��ծȯETF', '�ƽ�ETF', '��������ETF', '����ETF',
                   '��ҵ��ETF', '����500ETF', '����ETF', '��Դ����ETF', '��ɫETF', '����300ETF',
                   '��֤100ETF', '30��ծȯETF', '����˹ETF', '�ƽ�ETF'],
}


class A():
    pass


a = A()


def init(c):
    # �˻�
    c.account = text['�˻�']
    # �˻�����
    c.account_type = text['�˻�����']
    # ���׹�Ʊ��
    hold_limit = text['��������']
    a.trade_code_list = text['��Ʊ��']
    a.trade_code_name = text['��Ʊ������']
    c.run_time("run_tarder_func", "1nDay", "2024-07-25 09:45:00")
    c.run_time("run_tarder_func", "1nDay", "2024-07-25 14:45:00")
    c.run_time("reverse_repurchase_of_treasury_bonds_1", "1nDay", "2024-07-25 14:57:00")
    # 30����һ��
    # c.run_time("run_tarder_func","1800nSecond","2024-07-25 13:20:00")
    c.run_time("trader_info", "3nSecond", "2024-07-25 13:20:00")
    print(get_account(c, c.account, c.account_type))
    print(get_position(c, c.account, c.account_type))
    print(run_tarder_func(c))


def handlebar(c):
    # run_tarder_func(c)
    pass


def trader_info(c):
    if check_is_trader_date_1():
        print('{} �ȴ��������'.format(datetime.now()))
    else:
        print('{} ���ǽ���ʱ��ȴ��������'.format(datetime.now()))


def get_price(c, stock):
    '''
    ��ȡ���¼۸�
    '''
    tick = c.get_full_tick(stock_code=[stock])
    tick = tick[stock]
    price = tick['lastPrice']
    return price


def get_trader_stock(c):
    '''
    ��ȡ���׹�Ʊ��
    '''
    df = pd.DataFrame()
    try:
        df['֤ȯ����'] = text['��Ʊ��']
        df['����'] = text['��Ʊ������']
    except Exception as e:
        print(e, '��Ʊ�ػ�ȡ������')
    return df


def cacal_select_stock_factor(c):
    '''
    ����ѡ������
    '''
    user_factor = text['�Զ���ѡ�ɺ���']
    factor_name = list(user_factor.keys())
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
    df = get_trader_stock(c)
    if df.shape[0] > 0:
        df['�ֹɼ��'] = df['֤ȯ����'].apply(lambda x: '��' if x in hold_stock_list else '����')
        df = df[df['�ֹɼ��'] == '����']
    else:
        df
    if df.shape[0] > 0:
        user_factor_list = []
        stock_list = df['֤ȯ����'].tolist()
        for stock in stock_list:
            hist = c.get_market_data_ex([], stock_code=[stock], period="1d", count=-1,
                                        start_time='20210101',
                                        end_time='20500101',
                                        dividend_type='front')
            hist = hist[stock]
            tick = c.get_full_tick(stock_code=[stock])
            tick = tick[stock]
            result_list = []
            for name in factor_name:
                models = user_def_factor(hist, tick)
                try:
                    func = user_factor[name]
                    func = 'models.{}'.format(func)
                    result = eval(func)
                    result_list.append(result)
                except Exception as e:
                    print(e, '�����������')
                    result_list.append(None)
            user_factor_list.append(result_list)
        if len(user_factor_list[0]) < len(factor_name):
            df1 = pd.DataFrame()
        else:
            df1 = pd.DataFrame(user_factor_list)
            df1.columns = factor_name
            df1['֤ȯ����'] = stock_list
            df = pd.merge(df, df1, on=['֤ȯ����'])
    else:
        print('�Զ������Ӽ���û������')
    print('ѡ�����Ӽ���Ľ��888888888888')
    return df


def get_select_buy_stock(c):
    '''
    ѡ�����Ӽ���Ľ��
    '''
    df = cacal_select_stock_factor(c)
    user_factor = text['�Զ���ѡ������ѡ��']
    factor_name = list(user_factor.keys())
    select_select = []
    if df.shape[0] > 0:
        all_columns = df.columns.tolist()
        if len(factor_name) > 0:
            for name in factor_name:
                if name in all_columns:
                    try:
                        df[name] = pd.to_numeric(df[name])
                    except:
                        pass
                    try:
                        min_value = user_factor[name][0]
                        max_value = user_factor[name][-1]
                        df = df[df[name] >= min_value]
                        df = df[df[name] <= max_value]
                    except Exception as e:
                        print(e)
                        try:
                            factor_list = user_factor[name]
                            df['select'] = df[name].apply(lambda x: '��' if x in factor_list else '����')
                            df = df[df['select'] == '��']
                        except Exception as e:
                            print(e)
                            factor = user_factor[name]
                            df = df[df[name] == factor]
                    print('{}���ӷ������'.format(name))
                else:
                    print('{}���Ӳ������ӱ�����'.format(name))
        else:
            print('û���Զ�����������')
    else:
        print('û������Ĺ�Ʊ��******')
    return df


def cacal_sell_stock_factor(c):
    '''
    ������Ʊ���Ӽ���
    '''
    user_factor = text['�Զ�����������']
    factor_name = list(user_factor.keys())
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
    df = hold_stock
    if df.shape[0] > 0:
        user_factor_list = []
        stock_list = df['֤ȯ����'].tolist()
        for stock in stock_list:
            hist = c.get_market_data_ex([], stock_code=[stock], period="1d", count=-1,
                                        start_time='20210101',
                                        end_time='20500101',
                                        dividend_type='front')
            hist = hist[stock]
            tick = c.get_full_tick(stock_code=[stock])
            tick = tick[stock]
            result_list = []
            for name in factor_name:
                models = user_def_factor(hist, tick)
                try:
                    func = user_factor[name]
                    func = 'models.{}'.format(func)
                    result = eval(func)
                    result_list.append(result)
                except Exception as e:
                    print(e, '�����������')
                    result_list.append(None)
            user_factor_list.append(result_list)
        if len(user_factor_list[0]) < len(factor_name):
            df1 = pd.DataFrame()
        else:
            df1 = pd.DataFrame(user_factor_list)
            df1.columns = factor_name
            df1['֤ȯ����'] = stock_list
            df = pd.merge(df, df1, on=['֤ȯ����'])
    else:
        print('�Զ������Ӽ���û������')
    print('�������Ӽ���Ľ��888888888888')
    print(df)
    return df


def get_select_sell_stock(c):
    '''
    ѡ�����Ӽ���Ľ��
    '''
    df = cacal_sell_stock_factor(c)
    user_factor = text['�Զ�����������ѡ��']
    factor_name = list(user_factor.keys())
    or_df = pd.DataFrame()
    select_select = []
    df1 = df
    if df.shape[0] > 0:
        all_columns = df.columns.tolist()
        if len(factor_name) > 0:
            for name in factor_name:
                if name in all_columns:
                    try:
                        df[name] = pd.to_numeric(df[name])
                    except:
                        pass
                    try:
                        min_value = user_factor[name][0]
                        max_value = user_factor[name][-1]
                        df2 = df1[df1[name] >= min_value]
                        df2 = df2[df2[name] <= max_value]
                        or_df = pd.concat([or_df, df2], ignore_index=True)
                    except Exception as e:
                        print(e, 1, name)
                        try:
                            df = df1
                            factor_list = user_factor[name]
                            df['select'] = df[name].apply(lambda x: '��' if x in factor_list else '����')
                            df2 = df[df['select'] == '��']
                            or_df = pd.concat([or_df, df2], ignore_index=True)
                        except Exception as e:
                            print(e, 2, name)
                            factor = user_factor[name]
                            df2 = df1[df1[name] == factor]
                            or_df = pd.concat([or_df, df2], ignore_index=True)

                else:
                    print('{}���Ӳ������ӱ�����'.format(name))
            df = or_df
            df = df.drop_duplicates(keep='last')
            print(df)
        else:
            print('û���Զ�����������ѡ��')
    else:
        print('û����������************')
        df = pd.DataFrame()

    return df


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
    buy_df = get_select_buy_stock(c)
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
    sell_df = get_select_sell_stock(c)
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
            df = data
            is_del = text['�Ƿ�������']
            df['֤ȯ����'] = df['֤ȯ����'].astype(str)
            df['�������'] = df['֤ȯ����'].apply(lambda x: '��' if x in a.trade_code_list else '����')
            df = df[df['�������'] == '��']
            data = df

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


class user_def_factor:
    '''
    �Զ������ӿ��
    ������ͼٵ��ж���1��0���
    '''

    def __init__(self, df, tick):
        self.df = df
        self.tick = tick

    def cacal_limit(self):
        '''
        �����ǵ���
        '''
        try:
            tick = self.tick
            limit = ((tick['lastPrice'] - tick['lastClose']) / tick['lastClose']) * 100
            return limit
        except:
            print('{}�ǵ�������������'.format(stock))
            return -40

    def get_price(self):
        '''
        ��ȡ���¼۸�
        '''
        tick = self.tick
        price = tick['lastPrice']
        return price

    def mean_line_models(self, x1=3, x2=5, x3=10, x4=15, x5=20):
        '''
        ����ģ��
        ����ģ��
        5��10��20��30��60
        '''
        df = self.df
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

    def tarder_up_mean_line(self, line=5):
        '''
        վ�Ͻ��׾���
        '''

        df = self.df
        price = df['close'].tolist()[-1]
        df['line'] = df['close'].rolling(line).mean()
        mean_line = df['line'].tolist()[-1]
        if price >= mean_line:
            return 1
        else:
            return 0

    def tarder_down_mean_line(self, line=5):
        '''
        ���ƽ��׾���
        '''

        df = self.df
        price = df['close'].tolist()[-1]
        df['line'] = df['close'].rolling(line).mean()
        mean_line = df['line'].tolist()[-1]
        if price < mean_line:
            return 1
        else:
            return 0

    def stock_stop_up(self, limit=3):
        '''
        ����ֹӯ
        '''
        df = self.df

        df['day_limit'] = df['close'].pct_change() * 100
        day_limit = df['day_limit'].tolist()[-1]
        if limit >= day_limit:
            return 1
        else:
            return 0

    def stock_stop_up(self, limit=-6):
        '''
        ����ֹ��
        '''
        df = self.df

        df['day_limit'] = df['close'].pct_change() * 100
        day_limit = df['day_limit'].tolist()[-1]
        if day_limit <= limit:
            return 1
        else:
            return 0

    def six_pulse_excalibur(self):
        '''
        ������
        '''
        markers = 0
        signal = 0
        df = self.df
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
        return signal


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