# encoding:gbk
'''
����˹��Ƶ��ʱ�������ʵ��1
��ģ���̲���
����:����˹����
΢��:xg_quant
�޸��˻������Ϳ��Ըĳ��Լ���
"�˻�":"222",

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
import time
from datetime import datetime

pd.set_option('display.float_format', lambda x: '%.2f' % x)

text = {
    "�Զ��彻��Ʒ�ֽ���": "�Զ��彻�����ͱ����Ʊ����תծ��etf***********",
    "�˻�֧��������ȯ": "�˻�֧��������ȯ,�˻�����STOCK/CREDIT",
    "�˻�����": "STOCK",
    "�˻�": "55001948",
    "�˻�����": "STOCK",
    "�Ƿ����˵��": "ʵ�̸ĳɷ�",
    "�Ƿ����": "��",
    "�Ƿ�ʱ�����": "��",
    "����ʱ��": "20250425",
    "�Ƿ�������˵��": "����run_time���в��Ե�һ��ѭ�������µ������ǻ��¼����������Ҫ���",
    "�Ƿ�������": "��",
    "����۸����": 5,
    "�����۸����": 5,
    "�Ƿ�������": "��",
    "����ģʽ˵��": "����/���",
    "����ģʽ": "����",
    "�̶���������": 100,
    "������������": 300,
    "�̶����׽��": 200,
    "���н������": 400,
    "����ģʽ˵��": "�ٷֱ�/ATR/",
    "����ģʽ": "ATR",
    "�ٷֱ�ģʽ����": "�ٷֱ�ģʽ����",
    "������Ԫ��": 0.5,
    "���뵥Ԫ��": -0.5,

    "ATR��������": "ATR��������********",
    "ATR����": 14,
    "ATR��������": '1d',
    "�Ƿ��Զ�����ATR����": '��',
    "�Զ���̶�ATR����": "�Զ���̶�ATR����",
    "ATR����": 1,

    "ʱ������": "ʱ������********",
    "����ʱ���": 8,
    "���׿�ʼʱ��": 9,
    "���׽���ʱ��": 24,
    "�Ƿ�μӼ��Ͼ���": "��",
    "��ʼ���׷���": 0,
    "����Ʊ������": "����Ʊ������ �Զ���/�ֹ�",
    "����Ʊ��": "�Զ���",
    '�Զ����Ʊ��': "�Զ����Ʊ������",
    "��Ʊ������": "��������10�Ĺ�Ʊ������",
    "�Զ����Ʊ��": ['513100.SH', "159937.SZ", '511130.SH', '511090.SH',
                     '513400.SH', "512800.SH", '510300.SH', "515100.SH", '515450.SH',
                     "513500.SH"],
    "�Զ����Ʊ������": ['��˹���ETF', '�ƽ�ETF', '30���ծETF', '30���ծETF',
                         '����˹ETF', "����ETF", '����30ETF', '����ETF', '����ETF',
                         "����ETF"]

}


# ��¼��������
class A:
    pass


a = A()


def init(c):
    # �˻�
    c.account = text['�˻�']
    # �˻�����
    c.account_type = text['�˻�����']
    c.atr_preiod = text['ATR��������']
    if c.account_type == 'stock' or c.account_type == 'STOCK':
        c.buy_code = 23
        c.sell_code = 24
    else:
        # ������ȯ
        c.buy_code = 33
        c.sell_code = 34
    is_open = text['�Ƿ�������']
    if is_open == '��':
        a.del_log = True
    else:
        a.del_log = False
    c.buy_price_code = text['����۸����']
    c.sell_price_code = text['�����۸����']
    # ���׹�Ʊ��
    a.trade_code_list = text['�Զ����Ʊ��']
    a.trade_code_name = text['�Զ����Ʊ������']
    c.stock_name_dict = dict(zip(a.trade_code_list, a.trade_code_name))
    a.log = get_order_log(c)
    print('�������Խ��ף�������������������������')
    print('��ȡϵͳ��ί�����ݼ�¼��������һ������*******************************')
    print(a.log)
    print(get_account(c, c.account, c.account_type))
    print(get_position(c, c.account, c.account_type))
    # 3��һ��
    c.run_time("run_tarder_func", "3nSecond", "2024-07-25 13:20:00")
    # 1�����µ��˲��ɽ�����������
    c.run_time("run_order_trader_func", "60nSecond", "2024-07-25 13:20:00")


def handlebar(c):
    # run_tarder_func(c)
    pass


def get_order_log(c):
    '''
    ��һ�����л�ȡȫ��ί�б�ע���������Ͽ�û�м�¼����
    '''
    order = get_order(c, c.account, c.account_type)
    if order.shape[0] > 0:
        result_list = []
        order['Ͷ�ʱ�ע'] = order['Ͷ�ʱ�ע'].apply(lambda x: str(x).split(','))
        for j in order['Ͷ�ʱ�ע'].tolist():
            if len(j) == 7:
                result_list.append(j)
        if len(result_list) > 0:
            log = pd.DataFrame(result_list)
            log.columns = ['����', '֤ȯ����', '����ʱ��', '��������', '��������', '��������', '�����۸�']
            log['����ʱ��'] = log['����ʱ��'].apply(lambda x: int(''.join(str(x)[10:][:9].split(':'))))
        else:
            log = pd.DataFrame()
    else:
        log = pd.DataFrame()
    return log


def get_now_tick_data(c, stock='511090.SH'):
    '''
    ��ȡtick���ݵ���tick����
    '''
    test = text['�Ƿ�ʱ�����']
    test_date = text['����ʱ��']
    if test == '��':
        print('������������*************ʵ�̼ǵùر�{}'.format(test_date))
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
    ��������ʱ����,����ʵʱ����tick���ݼ���
    stock_type=�Զ���/�ֹ�
    'time'                  #ʱ���
    'lastPrice'             #���¼�
    'open'                  #���̼�
    'high'                  #��߼�
    'low'                   #��ͼ�
    'lastClose'             #ǰ���̼�
    'amount'                #�ɽ��ܶ�
    'volume'                #�ɽ�����
    'pvolume'               #ԭʼ�ɽ�����
    'stockStatus'           #֤ȯ״̬
    'openInt'               #�ֲ���
    'lastSettlementPrice'   #ǰ����
    'askPrice'              #ί����
    'bidPrice'              #ί���
    'askVol'                #ί����
    'bidVol'                #ί����
    'transactionNum'		#�ɽ�����
    '''
    name = '�ٷֱȸ�Ƶ��ʱ�������ʵ��'
    now_date = datetime.now()
    # '֤ȯ����','����ʱ��','�����ļ۸�','�ʽ�����','��������','��������','Ͷ�ʱ�ע'
    trader_date = str(datetime.now())[:10]
    # ʱ��ת����
    new_date = int(''.join(str(datetime.now())[10:][:9].split(':')))
    # ['֤ȯ����','����ʱ��','��������','��������','��������','�����۸�']
    log = a.log
    tick = get_now_tick_data(c, stock=stock)
    base_price = tick['lastClose'].tolist()[-1]
    price = tick['lastPrice'].tolist()[-1]
    grid_type = text['����ģʽ']
    stock_name = c.stock_name_dict.get(stock, stock)
    if grid_type == 'ATR':
        name = 'ATR��Ƶ��ʱ�������ʵ��'
        stock, price, atr_value, N, k, adjust_atr, atr_zdf = cacal_atr(c, stock)
        print(
            'ʱ��:{} ,��Ʊ:{},�۸�:{},atr:{} ,����:{} ,atr����:{} ,����atr:{} ,atr��Ӧ�ǵ���:{}'.format(datetime.now(),
                                                                                                        stock, price,
                                                                                                        atr_value, N, k,
                                                                                                        adjust_atr,
                                                                                                        atr_zdf))
        x1 = atr_zdf
        x2 = -atr_zdf
    else:
        name = '�ٷֱȸ�Ƶ��ʱ�������ʵ��'
    if log.shape[0] > 0:
        log['�����۸�'] = pd.to_numeric(log['�����۸�'])
        try:
            log['����ʱ��'] = log['����ʱ��'].apply(lambda x: int(''.join(str(x)[10:][:9].split(':'))))
        except:
            pass
        # print(log)
        log = log.sort_values(by='����ʱ��', ascending=True)
        log = log[log['֤ȯ����'] == stock]
        if log.shape[0] > 0:
            # �ϴν��״���������,�ϴδ�������������
            shift_trader_type = log['��������'].tolist()[-1]
            # ����ʱ��
            cf_time = log['����ʱ��'].tolist()[-1]
            # �ϴ��������´����룬�����˼������ǵ���û�д����´���������
            if shift_trader_type == 'sell':
                # �����������
                tick = tick[tick['date'] >= cf_time]
                # �����������ǵĵ�Ԫ��n,����
                pre_price = log['�����۸�'].tolist()[-1]
                n = ((price - pre_price) / pre_price) * 100
                ##�����������ǵĵ�Ԫ��n,����
                if n >= x1:
                    pre_price = log['�����۸�'].tolist()[-1]
                    print(
                        '{} {}���������������� Ŀǰ�����ǵ���{} ����Ŀǰ�ǵ���{}'.format(datetime.now(), stock, n, x1))
                else:
                    max_price = max(tick['lastPrice'].tolist())
                    max_price = max(max_price, pre_price)
                    print("{} {} Ŀǰ�۸�{} �ϴδ�����{} Ŀǰ�����ǵ���{} ��߼�{} �������ǵĳ���{}".format(
                        datetime.now(), stock, price, pre_price, n, max_price, max_price - pre_price))
                    pre_price = max_price
                    n = ((price - pre_price) / pre_price) * 100
            # �ϴδ�������,�´δ�������
            elif shift_trader_type == 'buy':
                # �����������
                tick = tick[tick['date'] >= cf_time]
                pre_price = log['�����۸�'].tolist()[-1]
                n = ((price - pre_price) / pre_price) * 100
                # �������������룬���������´���������
                if n <= x2:
                    pre_price = log['�����۸�'].tolist()[-1]
                    print(
                        '{} {}���������������� Ŀǰ�����ǵ���{} ����Ŀǰ�ǵ���{}'.format(datetime.now(), stock, n, x2))
                # û�д����´������������ڼ䲨��
                else:
                    min_price = min(tick['lastPrice'].tolist())
                    min_price = min(min_price, pre_price)
                    print("{} {} Ŀǰ�۸�{} �ϴδ�����{} Ŀǰ�����ǵ���{} ��ͼ�{} �����µ��ĳ���{}".format(
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
        print('{} ģ��{} ����{}  Ŀǰ�ǵ���{} ����Ŀǰ���ǵ���{} '.format(now_date, name, stock, zdf, x1))
        return name, 'sell'
    elif zdf <= x2:
        print('{} ģ��{} ����{}  Ŀǰ�ǵ���{} С��Ŀǰ���ǵ���{} '.format(now_date, name, stock, zdf, x2))
        return name, 'buy'
    else:
        print('{} ģ��{} �����Ͻ���{}  Ŀǰ�ǵ���{} Ŀǰ���ǵ���{}�����䲨�� '.format(now_date, name, stock, zdf, x1))
        return name, ''


def ATR(CLOSE, HIGH, LOW, N=14):
    '''
    ��ʵ����
    ���MTR:(��߼�-��ͼ�)��1��ǰ�����̼�-��߼۵ľ���ֵ�Ľϴ�ֵ��1��ǰ�����̼�-��ͼ۵ľ���ֵ�Ľϴ�ֵ
    �����ʵ����:MTR��N�ռ��ƶ�ƽ��
    '''
    MTR = MAX(MAX((HIGH - LOW), ABS(REF(CLOSE, 1) - HIGH)), ABS(REF(CLOSE, 1) - LOW))
    ATR = MA(MTR, N)
    return MTR, ATR


def cacal_auto_k_volatility(atr, lookback=30):
    '''
    ������ʷ�������Զ�����Kֵ
    '''
    # �������ATR������
    recent_atr = atr[-lookback:]
    atr_std = recent_atr.std()
    atr_mean = recent_atr.mean()

    # �����ʾ������
    volatility_ratio = atr_std / atr_mean

    # ���㸨��ָ��
    skewness = pd.Series(atr).skew()
    kurtosis = pd.Series(atr).kurt()

    # ��̬Kֵ����
    if volatility_ratio < 0.2:
        k = 0.8  # �Ͳ�������
    elif volatility_ratio < 0.5:
        k = 1.2  # ��������
    else:
        k = 1.8  # �߲�������
    return k


def cacal_atr(c, stock='513100.SH'):
    '''
    ����atr
    '''
    N = text['ATR����']
    is_open_auto_k = text['�Ƿ��Զ�����ATR����']
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
    if is_open_auto_k == '��':
        k = cacal_auto_k_volatility(atr, lookback=30)
    else:
        k = text['ATR����']
    atr_value = atr.tolist()[-1]
    adjust_atr = atr_value * k
    atr_zdf = (adjust_atr / price) * 100
    return stock, price, atr_value, N, k, adjust_atr, atr_zdf


def check_is_sell(c, accountid, datatype, stock='513100.SH', amount=100):
    '''
    ����Ƿ��������
    '''
    position = get_position(c, accountid, datatype)
    if position.shape[0] > 0:
        position = position[position['֤ȯ����'] == stock]
        if position.shape[0] > 0:
            position = position[position['�ֲ���'] >= 10]
            print(position)
            if position.shape[0] > 0:
                hold_amount = position['�ֲ���'].tolist()[-1]
                av_amount = position['��������'].tolist()[-1]
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
    ����Ƿ��������
    '''
    account = get_account(c, accountid, datatype)
    # ����ʹ�õ��ֽ�
    av_cash = account['���ý��']
    value = amount * price
    if av_cash >= value:
        return True
    else:
        return False


def check_hold_limit(c, accountid, datatype, stock='513100.SH', limit=1000):
    '''
    ����Ƿ񵽳ֹ�����
    '''
    position = get_position(c, accountid, datatype)
    if position.shape[0] > 0:
        position = position[position['֤ȯ����'] == stock]
        if position.shape[0] > 0:
            position = position[position['�ֲ���'] >= 10]
            if position.shape[0] > 0:
                hold_amount = position['�ֲ���'].tolist()[-1]
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
    ��������
    '''
    if stock[:3] in ['110', '113', '123', '127', '128', '111'] or stock[:2] in ['11', '12']:
        amount = math.floor(amount / 10) * 10
    else:
        amount = math.floor(amount / 100) * 100
    return amount


def run_tarder_func(c):
    '''
    ���н��׺���
    '''
    down_type = text['����ģʽ']
    fix_amount = text['�̶���������']
    hold_amount_limit = text['������������']
    fix_value = text['�̶����׽��']
    hold_value_lilit = text['���н������']
    stock_list_type = text['����Ʊ��']
    x1 = text['������Ԫ��']
    x2 = text['���뵥Ԫ��']
    test = text['�Ƿ����']
    if check_is_trader_date_1():
        if test == '��':
            print('��������ģʽʵ�̼ǵùر�*����������������������������������')
            a.log = pd.DataFrame()
        else:
            pass
        # print(a.log)
        now_date = datetime.now()
        if stock_list_type == '�Զ���':
            df = pd.DataFrame()
            df['֤ȯ����'] = text['�Զ����Ʊ��']
            df['֤ȯ����'] = text['�Զ����Ʊ������']
        else:
            df = get_position(c, c.account, c.account_type)
        if df.shape[0] > 0:
            for stock in df['֤ȯ����'].tolist():
                try:
                    # if True:
                    price = get_price(c, stock)
                    if down_type == '����':
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
                            print("{} {} ��������".format(datetime.now(), stock))
                            trader_type = ''
                            amount = fix_amount
                            price = price
                    elif trader_type == 'buy':
                        # ����Ƿ񵽴�ֹ�����
                        if check_hold_limit(c, c.account, c.account_type,
                                            stock=stock, limit=hold_amount_limit) == True:
                            # ����Ƿ��������
                            if check_is_buy(c, c.account, c.account_type, stock=stock, amount=fix_amount, price=price):

                                trader_type = 'buy'
                                amount = fix_amount
                                price = price
                            else:
                                trader_type = ''
                                amount = fix_amount
                                price = price
                                print("{} {} ���벻��".format(datetime.now(), stock))
                        else:
                            trader_type = ''
                            amount = fix_amount
                            price = price
                            print("{} {} �����볬����������".format(datetime.now(), stock))
                    else:
                        trader_type = ''
                        amount = fix_amount
                        price = price

                    if trader_type == 'buy' and amount >= 10:
                        # '֤ȯ����','����ʱ��','��������','��������','��������,'�����۸�''
                        flag = "{},{},{},{},{},{},{}".format(name, stock, now_date, 'buy', amount, hold_amount_limit,
                                                             price)
                        passorder(c.buy_code, 1101, c.account, str(stock), c.buy_price_code, 0, amount, flag, 1, flag,
                                  c)
                        print('{} {} {} ���¼۸�{} ����{} ����***************'.format(name, now_date, stock, price,
                                                                                      amount))
                    elif trader_type == 'sell' and amount >= 10:
                        flag = "{},{},{},{},{},{},{}".format(name, stock, now_date, 'sell', amount, hold_amount_limit,
                                                             price)
                        passorder(c.sell_code, 1101, c.account, str(stock), c.sell_price_code, 0, amount, flag, 1, flag,
                                  c)
                        print('{} {} {} ���¼۸�{} ����{} ����*******************'.format(name, now_date, stock, price,
                                                                                          amount))
                    else:
                        print('{} {} {} û�д�����������۲�'.format(name, now_date, stock))
                    if (trader_type == 'buy' or trader_type == 'sell') and amount >= 10:
                        # '֤ȯ����','����ʱ��','��������','��������','��������'
                        df1 = pd.DataFrame()
                        df1['����'] = [name]
                        df1['֤ȯ����'] = [stock]
                        df1['����ʱ��'] = [now_date]
                        df1['��������'] = [trader_type]
                        df1['��������'] = [amount]
                        df1['��������'] = [hold_amount_limit]
                        df1['�����۸�'] = [price]
                        df1['����ʱ��'] = df1['����ʱ��'].apply(lambda x: int(''.join(str(x)[10:][:9].split(':'))))
                        a.log = pd.concat([a.log, df1], ignore_index=True)
                    else:
                        pass
                # print(a.log)

                except Exception as e:
                    print(e, stock, '{}������������ܲ��ǽ�������'.format(datetime.now()))

            if a.del_log == True:
                print('��һ��ѭ����ս��׼�¼********************************')
                a.log = pd.DataFrame()
                a.del_log = False
            else:
                a.del_log = False

        else:
            print('{} ��ʱ�����Ʊû������'.format(now_date))
    else:
        print('{} ��ʱ�����Ʊ���ǽ���ʱ��'.format(datetime.now()))


def run_order_trader_func(c):
    '''
    �µ����ɽ��������µ�
    '''
    trader_log = get_order(c, c.account, c.account_type)
    now_date = str(datetime.now())[:10]
    # ���ɽ�����,ע��57����ǲ����µķϵ����������Ƿ���Ҫ
    not_list = [49, 50, 51, 52, 57]
    if trader_log.shape[0] > 0:
        trader_log['���ɽ�'] = trader_log['ί��״̬'].apply(lambda x: '��' if x in not_list else '����')
        trader_log = trader_log[trader_log['���ɽ�'] == '��']
    else:
        trader_log = trader_log
    name_list = ['ATR��Ƶ��ʱ�������ʵ��', "�ٷֱȸ�Ƶ��ʱ�������ʵ��"]
    try:
        trader_log = trader_log.drop_duplicates(subset=['Ͷ�ʱ�ע'], keep='last')
    except Exception as e:
        trader_log = pd.DataFrame()
        print(e)
    print('******************ί��')
    print(trader_log)
    if trader_log.shape[0] > 0:
        trader_log['֤ȯ����'] = trader_log['֤ȯ����'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
        trader_log['����'] = trader_log['Ͷ�ʱ�ע'].apply(lambda x: str(x).split(',')[0])
        trader_log['������'] = trader_log['����'].apply(lambda x: '��' if x in name_list else '����')
        trader_log = trader_log[trader_log['������'] == '��']
        if trader_log.shape[0] > 0:
            for stock, amount, trader_type, maker, oder_id, name in zip(trader_log['֤ȯ����'].tolist(),
                                                                        trader_log['δ�ɽ�����'].tolist(),
                                                                        trader_log['��������'].tolist(),
                                                                        trader_log['Ͷ�ʱ�ע'].tolist(),
                                                                        trader_log['�������'].tolist(),
                                                                        trader_log['����'].tolist()):
                price = get_price(c, stock)
                # δ�ɽ�����
                print('֤ȯ���룺{} δ�ɽ�����{}��������{} Ͷ�ʱ�ע{} ����id{}'.format(stock, amount, trader_type, maker,
                                                                                      oder_id))
                if trader_type == 49:
                    cancel(oder_id, c.account, c.account_type, c)
                    passorder(c.sell_code, 1101, c.account, str(stock), c.sell_price_code, 0, int(amount), str(maker),
                              1, str(maker), c)
                    print('���{} ���������������{} ����{} �۸�{}'.format(name, stock, amount, price))
                elif trader_type == 48:
                    cancel(oder_id, c.account, c.account_type, c)
                    passorder(c.buy_code, 1101, c.account, str(stock), c.buy_price_code, 0, int(amount), str(maker), 1,
                              str(maker), c)
                    print('���{} ��������������{} ����{} �۸�{}'.format(name, stock, amount, price))
                else:
                    print('���{} �������½���δ֪�Ľ�������'.format(name))
        else:
            print('���������µ����û��ί������')
    else:
        print('�����������µ�û��ί������')


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


def get_price(c, stock):
    '''
    ��ȡ���¼۸�
    '''
    tick = c.get_full_tick(stock_code=[stock])
    tick = tick[stock]
    price = tick['lastPrice']
    return price


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
            if is_del == '��':
                df['֤ȯ����'] = df['֤ȯ����'].astype(str)
                df['�������'] = df['֤ȯ����'].apply(lambda x: '��' if x in a.trade_code_list else '����')
                df = df[df['�������'] == '��']
                data = df
            else:
                data = data

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
            return 'sell', price, av_trader_amount
        else:
            print(
                '{} Ŀ������{} ��������{} ��������{}С�� ��������{} ����ȫ��'.format(stock, target_amount, hold_amount,
                                                                                     av_amount, av_trader_amount))
            return 'sell', price, av_amount
    else:
        print('{} Ŀ������{} ��������{}һ��������'.format(stock, target_amount, hold_amount))
        return '', '', ''


def RET(S, N=1):
    '''
    �������е�����N��ֵ,Ĭ�Ϸ������һ��
    '''
    return np.array(S)[-N]


def ABS(S):
    '''
    ����N�ľ���ֵ
    '''
    return np.abs(S)


def MAX(S1, S2):
    '''
    ����max
    '''
    return np.maximum(S1, S2)


def MIN(S1, S2):
    '''
    ����min
    '''
    return np.minimum(S1, S2)


def IF(S, A, B):
    '''
    ���в����ж� return=A  if S==True  else  B
    '''
    return np.where(S, A, B)


def REF(S, N=1):
    '''
    �������������ƶ�N,��������(shift������NAN)
    '''
    return pd.Series(S).shift(N).values


def MA(S, N):
    '''
    �����е�N�ռ��ƶ�ƽ��ֵ����������
    '''
    return pd.Series(S).rolling(N).mean().values