# encoding:gbk
'''
ȫ����������������������ֶ����Իز�2
���ߣ�С��
΢�ţ�15117320079
ʱ��:20250211
'''
import pandas as pd
import numpy as np
import talib


def init(c):
    # �˻�
    c.account = ''
    # �˻�����
    c.account_type = 'STOCK'
    # ��ʼʱ��
    c.start = '20200101 00:00:00'
    # ����ʱ��
    c.end = '20500101 00:00:00'
    # �����ʽ�
    c.capital = 300000
    c.stock_list = ['513100.SH', '511130.SH', '159937.SZ', '513350.SH',
                    '512890.SH', '159915.SZ', '513500.SH', '159985.SZ', '159981.SZ',
                    '159980.SZ', '513300.SH', '159680.SZ', '511090.SH', '513400.SH', '159934.SZ']
    c.name_list = ['��˹���ETF', '30��ծȯETF', '�ƽ�ETF', '��������ETF',
                   '����ETF', '��ҵ��ETF', '����500ETF', '����ETF', '��Դ����ETF',
                   '��ɫETF', '����300ETF', '��֤100ETF', '30��ծȯETF', '����˹ETF', '�ƽ�ETF']
    # ��������
    c.hold_limit = 10
    # �������
    c.buy_ratio = 0.1
    # ��������
    c.sell_ratio = 0
    # �������
    c.buy_score = 25
    # ��������
    c.sell_score = 0
    # ���׾���
    c.mean_line = 5
    # վ�Ͼ���
    c.up_lin = True
    # ���ƾ���
    c.down_line = True
    # ��������
    c.buy_n = 4
    # ��������
    c.sell_n = 3
    c.period_1 = '60m'
    c.df = pd.DataFrame()
    c.df['֤ȯ����'] = c.stock_list
    c.df['����'] = c.name_list
    print(c.df)
    # ������������
    # �ϰ汾�Ļز���Ҫ�趨��Ʊ��,�����ʷ����ʹ��
    c.set_universe(c.stock_list)


def handlebar(c):
    # ��ǰK��������
    d = c.barpos
    df = c.df
    # ��ȡ��ǰK������
    # ��ȷ������С����,��Ȼ��δ������
    today_1 = timetag_to_datetime(c.get_bar_timetag(d), '%Y%m%d%H%M%S')
    # ����
    # today_1=timetag_to_datetime(c.get_bar_timetag(d),'%Y-%m-%d)
    print(today_1)
    # ����ʹ��ǰ��Ȩ
    # hist=c.get_history_data(100,'1d',['open','close','low','high'],1)
    # �ֹ�
    hold_stock = get_position(c, c.account, c.account_type)
    account = get_account(c, c.account, c.account_type)
    if hold_stock.shape[0] > 0:
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
    df = c.df
    if df.shape[0] > 0:
        df['�ֹɼ��'] = df['֤ȯ����'].apply(lambda x: '��' if x in hold_stock_list else '����')
        df = df[df['�ֹɼ��'] == '����']
    else:
        df = df
    if df.shape[0] > 0:
        # ����ȫ������
        # ����
        score_list = []
        up_line_list = []
        n_list = []
        for stock in df['֤ȯ����'].tolist():
            hist = c.get_market_data_ex(
                fields=[],
                stock_code=[stock],
                period=c.period_1,
                start_time=str(c.start)[:8],
                end_time=today_1,
                count=-1,
                fill_data=True,
                subscribe=True)
            hist = hist[stock]
            close_list = hist['close'].tolist()
            signal, markers = six_pulse_excalibur(c, hist)
            n = signal.tolist()
            n_list.append(n)
            score = mean_line_models(c, close_list=close_list, x1=3, x2=5, x3=10, x4=15, x5=20)
            score_list.append(score)
            up_line = tarder_up_mean_line(c, close_list=close_list, line=c.mean_line)
            up_line_list.append(up_line)
        df['����'] = score_list
        df['����'] = n_list
        df['վ�Ͼ���'] = up_line_list
        buy_df = df[df['����'] >= c.buy_score]
        buy_df = buy_df[buy_df['����'] >= c.buy_n]
        buy_df = buy_df[buy_df['վ�Ͼ���'] == c.up_lin]

    else:
        buy_df = pd.DataFrame()
    # ��������
    if hold_stock.shape[0] > 0:
        score_list = []
        down_line_list = []
        n_list = []
        for stock in hold_stock['֤ȯ����'].tolist():
            hist = c.get_market_data_ex(
                fields=[],
                stock_code=[stock],
                period=c.period_1,
                start_time=str(c.start)[:8],
                end_time=today_1,
                count=-1,
                fill_data=True,
                subscribe=True)
            hist = hist[stock]
            close_list = hist['close'].tolist()
            signal, markers = six_pulse_excalibur(c, hist)
            n = signal.tolist()
            n_list.append(n)
            score = mean_line_models(c, close_list=close_list, x1=3, x2=5, x3=10, x4=15, x5=20)
            score_list.append(score)
            down_line = tarder_down_mean_line(c, close_list=close_list, line=c.mean_line)
            down_line_list.append(down_line)
        hold_stock['����'] = score_list
        hold_stock['����'] = n_list
        hold_stock['���ƾ���'] = down_line_list
        sell_df = hold_stock[hold_stock['����'] <= c.sell_score]
        sell_df = sell_df[sell_df['����'] <= c.sell_n]
        sell_df = sell_df[sell_df['���ƾ���'] == c.down_line]
    else:
        sell_df = pd.DataFrame()
    sell_amount = sell_df.shape[0]
    av_amount = (c.hold_limit - hold_amount) + sell_amount
    if av_amount < 0:
        print('����ֹ�����{}������'.format(c.hold_limit))
    else:
        av_amount = av_amount
    buy_df = buy_df[:av_amount]
    # ������
    print('{}�����Ʊ��******************'.format(today_1))
    print(buy_df)
    print('{}������Ʊ��******************'.format(today_1))
    print(sell_df)
    if sell_df.shape[0] > 0:
        for stock in sell_df['֤ȯ����'].tolist():
            order_target_percent(stock, c.sell_ratio, c, c.account)
            print('{} ������Ʊ{} '.format(today_1, stock))
    else:
        print('{} û�������Ĺ�Ʊ '.format(today_1))
    if buy_df.shape[0] > 0:
        for stock in buy_df['֤ȯ����'].tolist():
            order_target_percent(stock, c.buy_ratio, c, c.account)
            print('{} �����Ʊ{} '.format(today_1, stock))
    else:
        print('{} û������Ĺ�Ʊ '.format(today_1))


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



