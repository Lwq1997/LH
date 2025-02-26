# encoding:gbk
'''
全球大类低相关性六脉神剑趋势轮动策略回测2
作者：小果
微信：15117320079
时间:20250211
'''
import pandas as pd
import numpy as np
import talib


def init(c):
    # 账户
    c.account = ''
    # 账户类型
    c.account_type = 'STOCK'
    # 开始时间
    c.start = '20200101 00:00:00'
    # 结束时间
    c.end = '20500101 00:00:00'
    # 测试资金
    c.capital = 300000
    c.stock_list = ['513100.SH', '511130.SH', '159937.SZ', '513350.SH',
                    '512890.SH', '159915.SZ', '513500.SH', '159985.SZ', '159981.SZ',
                    '159980.SZ', '513300.SH', '159680.SZ', '511090.SH', '513400.SH', '159934.SZ']
    c.name_list = ['纳斯达克ETF', '30年债券ETF', '黄金ETF', '标普油气ETF',
                   '红利ETF', '创业板ETF', '标普500ETF', '豆粕ETF', '能源化工ETF',
                   '有色ETF', '沪深300ETF', '中证100ETF', '30年债券ETF', '道琼斯ETF', '黄金ETF']
    # 持有限制
    c.hold_limit = 10
    # 买入比率
    c.buy_ratio = 0.1
    # 卖出比率
    c.sell_ratio = 0
    # 买入分数
    c.buy_score = 25
    # 卖出分数
    c.sell_score = 0
    # 交易均线
    c.mean_line = 5
    # 站上均线
    c.up_lin = True
    # 跌破均线
    c.down_line = True
    # 买入周期
    c.buy_n = 4
    # 卖出周期
    c.sell_n = 3
    c.period_1 = '60m'
    c.df = pd.DataFrame()
    c.df['证券代码'] = c.stock_list
    c.df['名称'] = c.name_list
    print(c.df)
    # 动量因子天数
    # 老版本的回测需要设定股票池,配合历史数据使用
    c.set_universe(c.stock_list)


def handlebar(c):
    # 当前K线索引号
    d = c.barpos
    df = c.df
    # 获取当前K线日期
    # 精确到分钟小周期,不然有未来函数
    today_1 = timetag_to_datetime(c.get_bar_timetag(d), '%Y%m%d%H%M%S')
    # 日线
    # today_1=timetag_to_datetime(c.get_bar_timetag(d),'%Y-%m-%d)
    print(today_1)
    # 必须使用前复权
    # hist=c.get_history_data(100,'1d',['open','close','low','high'],1)
    # 持股
    hold_stock = get_position(c, c.account, c.account_type)
    account = get_account(c, c.account, c.account_type)
    if hold_stock.shape[0] > 0:
        hold_stock = hold_stock[hold_stock['持仓量'] >= 10]
        if hold_stock.shape[0] > 0:
            hold_stock_list = hold_stock['证券代码'].tolist()
            hold_amount = hold_stock.shape[0]
        else:
            hold_stock_list = []
            hold_amount = 0
    else:
        hold_stock_list = []
        hold_amount = 0
    df = c.df
    if df.shape[0] > 0:
        df['持股检查'] = df['证券代码'].apply(lambda x: '是' if x in hold_stock_list else '不是')
        df = df[df['持股检查'] == '不是']
    else:
        df = df
    if df.shape[0] > 0:
        # 计算全部因子
        # 分数
        score_list = []
        up_line_list = []
        n_list = []
        for stock in df['证券代码'].tolist():
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
        df['分数'] = score_list
        df['周期'] = n_list
        df['站上均线'] = up_line_list
        buy_df = df[df['分数'] >= c.buy_score]
        buy_df = buy_df[buy_df['周期'] >= c.buy_n]
        buy_df = buy_df[buy_df['站上均线'] == c.up_lin]

    else:
        buy_df = pd.DataFrame()
    # 卖出分析
    if hold_stock.shape[0] > 0:
        score_list = []
        down_line_list = []
        n_list = []
        for stock in hold_stock['证券代码'].tolist():
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
        hold_stock['分数'] = score_list
        hold_stock['周期'] = n_list
        hold_stock['跌破均线'] = down_line_list
        sell_df = hold_stock[hold_stock['分数'] <= c.sell_score]
        sell_df = sell_df[sell_df['周期'] <= c.sell_n]
        sell_df = sell_df[sell_df['跌破均线'] == c.down_line]
    else:
        sell_df = pd.DataFrame()
    sell_amount = sell_df.shape[0]
    av_amount = (c.hold_limit - hold_amount) + sell_amount
    if av_amount < 0:
        print('到达持股限制{}不买入'.format(c.hold_limit))
    else:
        av_amount = av_amount
    buy_df = buy_df[:av_amount]
    # 先卖出
    print('{}买入股票池******************'.format(today_1))
    print(buy_df)
    print('{}卖出股票池******************'.format(today_1))
    print(sell_df)
    if sell_df.shape[0] > 0:
        for stock in sell_df['证券代码'].tolist():
            order_target_percent(stock, c.sell_ratio, c, c.account)
            print('{} 卖出股票{} '.format(today_1, stock))
    else:
        print('{} 没有卖出的股票 '.format(today_1))
    if buy_df.shape[0] > 0:
        for stock in buy_df['证券代码'].tolist():
            order_target_percent(stock, c.buy_ratio, c, c.account)
            print('{} 买入股票{} '.format(today_1, stock))
    else:
        print('{} 没有买入的股票 '.format(today_1))


def six_pulse_excalibur(c, df):
    '''
    六脉神剑
    '''
    markers = 0
    signal = 0
    # df=self.data.get_hist_data_em(stock=stock)
    CLOSE = df['close']
    LOW = df['low']
    HIGH = df['high']
    DIFF = EMA(CLOSE, 8) - EMA(CLOSE, 13)
    DEA = EMA(DIFF, 5)
    # 如果满足DIFF>DEA 在1的位置标记1的图标
    # DRAWICON(DIFF>DEA,1,1);
    markers += IF(DIFF > DEA, 1, 0)[-1]
    # 如果满足DIFF<DEA 在1的位置标记2的图标
    # DRAWICON(DIFF<DEA,1,2);
    markers += IF(DIFF < DEA, 1, 0)[-1]
    # DRAWTEXT(ISLASTBAR=1,1,'. MACD'),COLORFFFFFF;{微信公众号:数据分析与运用}
    ABC1 = DIFF > DEA
    signal += IF(ABC1, 1, 0)[-1]
    尊重市场1 = (CLOSE - LLV(LOW, 8)) / (HHV(HIGH, 8) - LLV(LOW, 8)) * 100
    K = SMA(尊重市场1, 3, 1)
    D = SMA(K, 3, 1)
    # 如果满足k>d 在2的位置标记1的图标
    markers += IF(K > D, 1, 0)[-1]
    # DRAWICON(K>D,2,1);
    markers += IF(K < D, 1, 0)[-1]
    # DRAWICON(K<D,2,2);
    # DRAWTEXT(ISLASTBAR=1,2,'. KDJ'),COLORFFFFFF;
    ABC2 = K > D
    signal += IF(ABC2, 1, 0)[-1]
    指标营地 = REF(CLOSE, 1)
    RSI1 = (SMA(MAX(CLOSE - 指标营地, 0), 5, 1)) / (SMA(ABS(CLOSE - 指标营地), 5, 1)) * 100
    RSI2 = (SMA(MAX(CLOSE - 指标营地, 0), 13, 1)) / (SMA(ABS(CLOSE - 指标营地), 13, 1)) * 100
    markers += IF(RSI1 > RSI2, 1, 0)[-1]
    # DRAWICON(RSI1>RSI2,3,1);
    markers += IF(RSI1 < RSI2, 1, 0)[-1]
    # DRAWICON(RSI1<RSI2,3,2);
    # DRAWTEXT(ISLASTBAR=1,3,'. RSI'),COLORFFFFFF;
    ABC3 = RSI1 > RSI2
    signal += IF(ABC3, 1, 0)[-1]
    尊重市场 = -(HHV(HIGH, 13) - CLOSE) / (HHV(HIGH, 13) - LLV(LOW, 13)) * 100
    LWR1 = SMA(尊重市场, 3, 1)
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
    均线模型
    趋势模型
    5，10，20，30，60
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
    # 加分的情况
    mean_x1 = df1['x1'].tolist()[-1]
    mean_x2 = df1['x2'].tolist()[-1]
    mean_x3 = df1['x3'].tolist()[-1]
    mean_x4 = df1['x4'].tolist()[-1]
    mean_x5 = df1['x5'].tolist()[-1]
    # 相邻2个均线进行比较
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
    站上交易均线
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
    跌破交易均线
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
    当日止盈
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
    当日止损
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
    if len(positions) > 0:
        df = pd.DataFrame()
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
    else:
        data = pd.DataFrame()
    return data


def RD(N, D=3):
    # 四舍五入取3位小数
    return np.round(N, D)


def RET(S, N=1):
    # 返回序列倒数第N个值,默认返回最后一个
    return np.array(S)[-N]


def ABS(S
        ):
    # 返回N的绝对值
    return np.abs(S)


def MAX(S1, S2):
    # 序列max
    return np.maximum(S1, S2)


def MIN(S1, S2):
    # 序列min
    return np.minimum(S1, S2)


def IF(S, A, B):
    # 序列布尔判断 return=A  if S==True  else  B
    return np.where(S, A, B)


def AND(S1, S2):
    # and
    return np.logical_and(S1, S2)


def OR(S1, S2):
    # or
    return np.logical_or(S1, S2)


def RANGE(A, B, C):
    '''
    期间函数
    B<=A<=C
    '''
    df = pd.DataFrame()
    df['select'] = A.tolist()
    df['select'] = df['select'].apply(lambda x: True if (x >= B and x <= C) else False)
    return df['select']


def REF(S, N=1):  # 对序列整体下移动N,返回序列(shift后会产生NAN)
    return pd.Series(S).shift(N).values


def DIFF(S, N=1):  # 前一个值减后一个值,前面会产生nan
    return pd.Series(S).diff(N).values  # np.diff(S)直接删除nan，会少一行


def STD(S, N):  # 求序列的N日标准差，返回序列
    return pd.Series(S).rolling(N).std(ddof=0).values


def SUM(S, N):  # 对序列求N天累计和，返回序列    N=0对序列所有依次求和
    return pd.Series(S).rolling(N).sum().values if N > 0 else pd.Series(S).cumsum().values


def CONST(S):  # 返回序列S最后的值组成常量序列
    return np.full(len(S), S[-1])


def HHV(S, N):  # HHV(C, 5) 最近5天收盘最高价
    return pd.Series(S).rolling(N).max().values


def LLV(S, N):  # LLV(C, 5) 最近5天收盘最低价
    return pd.Series(S).rolling(N).min().values


def HHVBARS(S, N):  # 求N周期内S最高值到当前周期数, 返回序列
    return pd.Series(S).rolling(N).apply(lambda x: np.argmax(x[::-1]), raw=True).values


def LLVBARS(S, N):  # 求N周期内S最低值到当前周期数, 返回序列
    return pd.Series(S).rolling(N).apply(lambda x: np.argmin(x[::-1]), raw=True).values


def MA(S, N):  # 求序列的N日简单移动平均值，返回序列
    return pd.Series(S).rolling(N).mean().values


def EMA(S, N):  # 指数移动平均,为了精度 S>4*N  EMA至少需要120周期     alpha=2/(span+1)
    return pd.Series(S).ewm(span=N, adjust=False).mean().values


def SMA(S, N, M=1):  # 中国式的SMA,至少需要120周期才精确 (雪球180周期)    alpha=1/(1+com)
    return pd.Series(S).ewm(alpha=M / N, adjust=False).mean().values  # com=N-M/M


def DMA(S, A):  # 求S的动态移动平均，A作平滑因子,必须 0<A<1  (此为核心函数，非指标）
    return pd.Series(S).ewm(alpha=A, adjust=True).mean().values


def WMA(S, N):  # 通达信S序列的N日加权移动平均 Yn = (1*X1+2*X2+3*X3+...+n*Xn)/(1+2+3+...+Xn)
    return pd.Series(S).rolling(N).apply(lambda x: x[::-1].cumsum().sum() * 2 / N / (N + 1), raw=True).values


def AVEDEV(S, N):  # 平均绝对偏差  (序列与其平均值的绝对差的平均值)
    return pd.Series(S).rolling(N).apply(lambda x: (np.abs(x - x.mean())).mean()).values


def SLOPE(S, N):  # 返S序列N周期回线性回归斜率
    return pd.Series(S).rolling(N).apply(lambda x: np.polyfit(range(N), x, deg=1)[0], raw=True).values


def FORCAST(S, N):  # 返回S序列N周期回线性回归后的预测值， jqz1226改进成序列出
    return pd.Series(S).rolling(N).apply(lambda x: np.polyval(np.polyfit(range(N), x, deg=1), N - 1), raw=True).values


def LAST(S, A, B):  # 从前A日到前B日一直满足S_BOOL条件, 要求A>B & A>0 & B>=0
    return np.array(pd.Series(S).rolling(A + 1).apply(lambda x: np.all(x[::-1][B:]), raw=True), dtype=bool)


# ------------------   1级：应用层函数(通过0级核心函数实现） ----------------------------------
def COUNT(S, N):  # COUNT(CLOSE>O, N):  最近N天满足S_BOO的天数  True的天数
    return SUM(S, N)


def EVERY(S, N):  # EVERY(CLOSE>O, 5)   最近N天是否都是True
    return IF(SUM(S, N) == N, True, False)


def EXIST(S, N):  # EXIST(CLOSE>3010, N=5)  n日内是否存在一天大于3000点
    return IF(SUM(S, N) > 0, True, False)


def FILTER(S, N):  # FILTER函数，S满足条件后，将其后N周期内的数据置为0, FILTER(C==H,5)
    for i in range(len(S)): S[i + 1:i + 1 + N] = 0 if S[i] else S[i + 1:i + 1 + N]
    return S  # 例：FILTER(C==H,5) 涨停后，后5天不再发出信号


def BARSLAST(S):  # 上一次条件成立到当前的周期, BARSLAST(C/REF(C,1)>=1.1) 上一次涨停到今天的天数
    M = np.concatenate(([0], np.where(S, 1, 0)))
    for i in range(1, len(M)):  M[i] = 0 if M[i] else M[i - 1] + 1
    return M[1:]


def BARSLASTCOUNT(S):  # 统计连续满足S条件的周期数        by jqz1226
    rt = np.zeros(len(S) + 1)  # BARSLASTCOUNT(CLOSE>OPEN)表示统计连续收阳的周期数
    for i in range(len(S)): rt[i + 1] = rt[i] + 1 if S[i] else rt[i + 1]
    return rt[1:]


def BARSSINCEN(S, N):  # N周期内第一次S条件成立到现在的周期数,N为常量  by jqz1226
    return pd.Series(S).rolling(N).apply(lambda x: N - 1 - np.argmax(x) if np.argmax(x) or x[0] else 0,
                                         raw=True).fillna(0).values.astype(int)


def CROSS(S1, S2):  # 判断向上金叉穿越 CROSS(MA(C,5),MA(C,10))  判断向下死叉穿越 CROSS(MA(C,10),MA(C,5))
    return np.concatenate(([False], np.logical_not((S1 > S2)[:-1]) & (S1 > S2)[1:]))  # 不使用0级函数,移植方便  by jqz1226


def CROSS_UP(S1, S2):  # 判断向上金叉穿越 CROSS(MA(C,5),MA(C,10))  判断向下死叉穿越 CROSS(MA(C,10),MA(C,5))
    return np.concatenate(([False], np.logical_not((S1 > S2)[:-1]) & (S1 > S2)[1:]))  # 不使用0级函数,移植方便  by jqz1226


def CROSS_DOWN(S1, S2):
    return np.concatenate(([False], np.logical_not((S1 < S2)[:-1]) & (S1 < S2)[1:]))  # 不使用0级函数,移植方便  by jqz1226


def LONGCROSS(S1, S2, N):  # 两条线维持一定周期后交叉,S1在N周期内都小于S2,本周期从S1下方向上穿过S2时返回1,否则返回0
    return np.array(np.logical_and(LAST(S1 < S2, N, 1), (S1 > S2)), dtype=bool)  # N=1时等同于CROSS(S1, S2)


def VALUEWHEN(S, X):  # 当S条件成立时,取X的当前值,否则取VALUEWHEN的上个成立时的X值   by jqz1226
    return pd.Series(np.where(S, X, np.nan)).ffill().values



