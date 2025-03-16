# encoding:gbk
'''
六脉神剑趋势轮动策略
同花顺人气etf排行，自动全市场算法选股交易
修改这个参数
"账户":"",
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
    "自定义交易品种交易": "自定义交易类型比如股票，可转债，etf***********",
    "账户": "",
    "账户类型": "STOCK",
    "是否隔离策略": "否",
    "交易模式说明": "金额/数量",
    "交易模式": "金额",
    "固定交易金额": 10000,
    "固定交易数量": 100,
    "特殊交易标的设置": "特殊交易标的设置",
    "特殊交易标的": ['511360.SH', '159651.SZ', '511580.SH', '511380.SH', '159649', '511270.SH',
                     '511030.SH', '511100.SH', '159816.SZ', '159651.SZ', '159972.SZ', '159651.SZ', '511260.SH',
                     '511010.SH', '511220.SH', '511020.SH', '511520.SH', '511060.SH', '511180.SH', '511130.SH',
                     '511090.SH'],
    "特殊交易标的固定交易金额": 15000,
    "特殊交易标的固定交易数量": 100,
    "趋势设置": "趋势设置",
    "是否开启均线趋势买入": "是",
    "价格站上N日线买入": 5,
    "是否自定义交易品种跌破N日均线卖出": "是",
    "自定义交易品种跌破N日均线卖出": 5,
    "自定义交易品种持有分数": 50,
    "买入最低分": 50,
    "买入前N": 10,
    "持有限制": 10,
    "持股限制": 10,
    "买入交易涨跌幅设置": "交易涨跌幅设置",
    "最大涨跌幅": 6,
    "最小涨跌幅": -3,
    "六脉神剑设置": "自定义六脉神剑设置*****************",
    "是否开启周周期说明": "短线可以不，快进快出，趋势可以开启",
    "是否开启周周期": "否",
    "周周期买入": 2,
    "周周期清仓": 1,
    "是否开启日周期": "是",
    "日周期买入": 4,
    "日周期清仓": 3,
    "是否开启大涨卖出": "是",
    "大涨": 3,
    "是否开启大跌不卖": "否",
    "大跌": -1.5,
    "时间设置": "时间设置********",
    "交易时间段": 4,
    "交易开始时间": 9,
    "交易结束时间": 14,
    "是否参加集合竞价": "否",
    "开始交易分钟": 0,
    "分钟脉冲设置": "分钟脉冲设置",
    "脉冲时间": 10,
    "向上脉冲": 2,
    "向下脉冲": -40,
    '自定义股票池': "自定义股票池设置",
    "服务器": "http://124.220.32.224",
    "端口": "8888",
    "授权码": "xg123456",
}
'''
自定义股票池格式
证券代码       名称
513100.SH     纳斯达克ETF
159502.SZ     标普生物ETF
特别注意证券代码要带市场.SZ,.SH
'''


class A():
    pass


a = A()


class xg_data:
    '''
    小果数据api，支持qmt,本地
    '''

    def __init__(self, url='http://124.220.32.224', port=8888, password='123456'):
        '''
        小果数据api，支持qmt,本地
        url服务器网页
        port端口
        password授权码
        '''
        self.url = url
        self.port = port
        self.password = password

    def get_user_info(self):
        '''
        获取用户信息
        '''
        url = '{}:{}/_dash-update-component'.format(self.url, self.port)
        headers = {'Content-Type': 'application/json'}
        data = {
            "output": "finace_data_table_1.data@e60ed22f488acd1653d4a92a187c4775d06cc39e4afa58da3bee9c8261dcc6a0",
            "outputs": {"id": "finace_data_table_1",
                        "property": "data@e60ed22f488acd1653d4a92a187c4775d06cc39e4afa58da3bee9c8261dcc6a0"},
            "inputs": [{"id": "finace_data_password", "property": "value", "value": self.password},
                       {"id": "finace_data_data_type", "property": "value", "value": "代码"},
                       {"id": "finace_data_text", "property": "value",
                        "value": "from trader_tool.stock_data import stock_data\nstock_data=stock_data()\ndf=stock_data.get_stock_hist_data_em(stock='600031',start_date='20210101',end_date='20600101',data_type='D',count=8000)\ndf.to_csv(r'{}\\数据\\{}数据.csv')\n                \n                "},
                       {"id": "finace_data_run", "property": "value", "value": "运行"},
                       {"id": "finace_data_down_data", "property": "value", "value": "不下载数据"}],
            "changedPropIds": ["finace_data_run.value"], "parsedChangedPropsIds": ["finace_data_run.value"]}

        res = requests.post(url=url, data=json.dumps(data), headers=headers)
        text = res.json()
        df = pd.DataFrame(text['response']['finace_data_table_1']['data'])
        return df

    def get_user_def_data(self, func=''):
        '''
        自定义数据获取
        调用数据库
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
                       {"id": "finace_data_data_type", "property": "value", "value": "代码"},
                       {"id": "finace_data_text", "property": "value", "value": func},
                       {"id": "finace_data_run", "property": "value", "value": "运行"},
                       {"id": "finace_data_down_data", "property": "value", "value": "不下载数据"}],
            "changedPropIds": ["finace_data_run.value"], "parsedChangedPropsIds": ["finace_data_run.value"]}
        res = requests.post(url=url, data=json.dumps(data), headers=headers)
        text = res.json()
        df = pd.DataFrame(text['response']['finace_data_table']['data'])
        return info, df

    def params_func(self, text=''):
        '''
        解析函数
        '''
        data_list = []
        f = text.split('\n')
        for i in f:
            text = i.strip().lstrip()
            data_list.append(text)
        func = '\n'.join(data_list)
        return func


def init(c):
    # 账户
    c.account = text['账户']
    # 账户类型
    c.account_type = text['账户类型']
    # 交易股票池
    hold_limit = text['持有限制']
    # 读取交易股票池
    c.url = text['服务器']
    c.port = text['端口']
    c.user_password = text['授权码']
    print('小果服务器提高数据支持')
    print('服务器{}'.format(c.url))
    print('端口{}'.format(c.port))
    print('授权码{}'.format(c.user_password))
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
    读取交易股票池
    '''
    func = '''
		from trader_tool.ths_rq import ths_rq
		rq=ths_rq()
		df=rq.get_etf_hot_rank()
		print(df)
		'''
    info, df = c.data.get_user_def_data(func=func)
    print(df)
    stats = df['数据状态'].tolist()[-1]
    if stats == True:
        df['证券代码'] = df['代码']
        df['证券代码'] = df['证券代码'].apply(lambda x: str(x) + '.SH' if str(x)[:2] == '51' else str(x) + '.SZ')
        df['基金名称'] = df['名称']
        df['标记1'] = df['基金名称'].apply(lambda x: str(x)[:2])
        df['标记2'] = df['基金名称'].apply(lambda x: str(x).split('ETF')[0][-2:])
        df = df.drop_duplicates(subset=['标记1'])
        df = df.drop_duplicates(subset=['标记2'])
        print(df)
    else:
        df = df
    return df


def read_tdx_trader_stock(c, path=r'C:\new_tdx\T0002\blocknew\BUY.blk'):
    '''
    读取通达信板块自选股交易
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
        df['证券代码'] = stock_list
        df['证券代码'] = df['证券代码'].apply(
            lambda x: str(x)[-6:] + '.SH' if str(x)[0] == '1' else str(x)[-6:] + '.SZ')
        return df
    except:
        print('路径有问题{}'.format(path))
        df = pd.DataFrame()
        return df


def get_mean_line_analyis(c):
    '''
    交易均线分析
    '''
    df = read_trader_stock(c)
    line = text['价格站上N日线买入']

    hold_stock = get_position(c, c.account, c.account_type)
    if hold_stock.shape[0] > 0:
        hold_stock = hold_stock[hold_stock['持仓量'] >= 10]
        if hold_stock.shape[0] > 0:
            hold_stock_list = hold_stock['证券代码'].tolist()
        else:
            hold_stock_list = []
    else:
        hold_stock_list = []

    df['持股检查'] = df['证券代码'].apply(lambda x: '是' if x in hold_stock_list else '不是')
    df = df[df['持股检查'] == '不是']
    print(df)
    select_list = []
    if df.shape[0] > 0:
        for stock in df['证券代码'].tolist():

            try:
                hist = c.get_market_data_ex([], stock_code=[stock], period="1d", count=-1,
                                            start_time='20210101',
                                            end_time='20500101',
                                            dividend_type='front')
                hist = hist[stock]
                stats = tarder_up_mean_line(c, close_list=hist['close'].tolist(), line=line)
                if stats == True:
                    select_list.append('是')
                else:
                    select_list.append('不是')
            except:
                print('交易均线分析{} 计算有问题'.format(stock))
                select_list.append('不是')
        df['交易均线分析'] = select_list
        df = df[df['交易均线分析'] == '是']
        return df
    else:
        return df


def get_score_analysis(c):
    '''
    分数分析
    '''
    print('分数分析*************')
    min_score = text['买入最低分']
    df = get_mean_line_analyis(c)
    score_list = []
    if df.shape[0] > 0:
        for stock in df['证券代码'].tolist():
            try:
                hist = c.get_market_data_ex([], stock_code=[stock], period="1d", count=-1,
                                            start_time='20210101',
                                            end_time='20500101',
                                            dividend_type='front')
                hist = hist[stock]
                score = mean_line_models(c, close_list=hist['close'].tolist())
                score_list.append(score)
            except:
                print('{} 分数计算有问题'.format(stock))
                score_list.append(0)
        df['分数'] = score_list
        df = df[df['分数'] >= min_score]
        return df
    else:
        return df


def get_limit_analysis(c):
    '''
    涨跌幅分析
    '''
    print('涨跌幅分析****************')
    df = get_score_analysis(c)
    min_limit = text['最小涨跌幅']
    max_limit = text['最大涨跌幅']
    limit_list = []
    if df.shape[0] > 0:
        for stock in df['证券代码'].tolist():
            limit = cacal_limit(c, stock)
            limit_list.append(limit)
        df['涨跌幅'] = limit_list
        df = df[df['涨跌幅'] >= min_limit]
        df = df[df['涨跌幅'] <= max_limit]
        return df
    else:
        return df


def cacal_cycle_analysis(c):
    '''
    计算周周期
    '''
    print('计算周周期************')
    min_ratio = text['周周期买入']
    is_open = text['是否开启周周期']
    df = get_limit_analysis(c)
    if is_open == '是':
        print('开启周周期&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        if df.shape[0] > 0:
            cycle_list = []
            df['证券代码'] = df['证券代码'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
            for stock in df['证券代码'].tolist():

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
            df['周周期'] = cycle_list
            df = df[df['周周期'] >= min_ratio]
            return df
        else:
            df = pd.DataFrame()
            return df
    else:
        print('不开启周周期&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        return df


def cacal_diurnal_cycle(c):
    '''
    计算日周期
    '''
    print("计算日周期*******************")
    min_ratio = text['日周期买入']
    is_open = text['是否开启日周期']
    df = cacal_cycle_analysis(c)
    if is_open == '是':
        print('开启日周期&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&')
        if df.shape[0] > 0:
            cycle_list = []
            df['证券代码'] = df['证券代码'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
            for stock in df['证券代码'].tolist():
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
            df['日周期'] = cycle_list
            print('日周期************')
            print(df)
            df = df[df['日周期'] >= min_ratio]

            return df
        else:
            df = pd.DataFrame()
            return df
    else:
        return df


def get_sell_stock_data(c):
    '''
    获取卖出股票数据
    '''
    print('获取卖出股票数据***********')
    is_del = text['是否隔离策略']
    is_open_down_mean_line = text['是否自定义交易品种跌破N日均线卖出']
    men_line_n = text['自定义交易品种跌破N日均线卖出']
    is_open_week_n_sell = text['是否开启周周期']
    week_n_sell = text['周周期清仓']
    daily_n_sell = text['日周期清仓']
    zdf_sell = text['大涨']
    is_open_zdf_not_sell = text['是否开启大跌不卖']
    zdf_not_sell = text['大跌']
    is_open_max_zdf_sell = text['是否开启大涨卖出']
    hold_score = text['自定义交易品种持有分数']
    df = get_position(c, c.account, c.account_type)
    print("隔离策略***************************")
    trader_df = read_trader_stock(c)
    if trader_df.shape[0] > 0:
        trader_df['证券代码'] = trader_df['证券代码'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
        trader_stock_list = trader_df['证券代码'].tolist()

    else:
        trader_stock_list = []
    if is_del == '是':
        print('隔离策略**************')
        if df.shape[0] > 0:
            df['证券代码'] = df['证券代码'].astype(str)
            df['隔离策略'] = df['证券代码'].apply(lambda x: '是' if x in trader_stock_list else '不是')
            df1 = df[df['隔离策略'] == '不是']
            df = df[df['隔离策略'] == '是']
            print('自定义股票池标的************************')
            print(df)
            print('不是自定义交易股票的标的*******************')
            print(df1)
        else:
            df = df
    if df.shape[0] > 0:
        week_cycle_list = []
        for stock in df['证券代码'].tolist():
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
        df['周周期'] = week_cycle_list
        daily_cycle_list = []
        for stock in df['证券代码'].tolist():
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
        df['日周期'] = daily_cycle_list
        if is_open_down_mean_line == '是':
            print('开启均线分析********')
            down_list = []
            for stock in df['证券代码'].tolist():
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
                    down_list.append('是')
            df['均线分析'] = down_list
        else:
            df['均线分析'] = '不是'
        if is_open_max_zdf_sell == '是':
            zdf_list = []
            for stock in df['证券代码'].tolist():
                try:
                    zdf = cacal_limit(c, stock)
                    zdf_list.append(zdf)
                except Exception as e:
                    print(e)
                    zdf_list.append(-40)
            df['涨跌幅'] = zdf_list
        else:
            df['涨跌幅'] = -40
        mean_score_list = []
        for stock in df['证券代码'].tolist():
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
                print('持股均线计算有问题{}', format(stock))
                mean_score_list.append(100)
        df['趋势得分'] = mean_score_list
        sell_stock_list = []
        print('(((((((((((((((持股分析((((((((((((((((9')
        print(df)
        for stock, week_cycle, daily_cycle, score, down, zdf in zip(df['证券代码'],
                                                                    df['周周期'], df['日周期'], df['趋势得分'],
                                                                    df['均线分析'], df['涨跌幅']):
            if is_open_week_n_sell == '是' and week_cycle <= week_n_sell:
                print('{} 周周期{} 小于平仓周周期{} 平仓'.format(stock, week_cycle, week_cycle))
                sell_stock_list.append(stock)
            elif daily_cycle <= daily_n_sell:
                print('{} 日周期{} 小于平仓日周期{} 平仓'.format(stock, daily_cycle, daily_n_sell))
                sell_stock_list.append(stock)
            elif score < hold_score:
                print('{} 分数{} 小于持有分数{} 平仓'.format(stock, score, hold_score))
                sell_stock_list.append(stock)
            elif down == True:
                print('{} 跌破均线{} 平仓'.format(stock, men_line_n))
                sell_stock_list.append(stock)
            elif zdf >= zdf_sell:
                print('{} 涨跌幅{} 大于平仓涨跌幅{} 平仓'.format(stock, zdf, zdf_sell))
                sell_stock_list.append(stock)
            else:
                print('{} 不符合卖出模型继续持有'.format(stock))
            sell_stock_list = list(set(sell_stock_list))
            for stock, zdf in zip(df['证券代码'], df['涨跌幅']):
                if zdf <= zdf_not_sell:
                    if is_open_zdf_not_sell == '是':
                        try:
                            sell_stock_list.remove(stock)
                            print('{}开启大跌不卖 涨跌幅{} 小于大跌涨跌幅'.format(stock, zdf, zdf_not_sell))
                        except:
                            pass
                    else:
                        pass
                else:
                    pass
        df['可用持股'] = df['证券代码'].apply(lambda x: '是' if x in sell_stock_list else '不是')
        df = df[df['可用持股'] == '是']
        return df
    else:
        sell_df = pd.DataFrame()
        return sell_df


def get_buy_sell_stock_data(c):
    '''
    获取买卖数据
    '''
    print('获取买卖数据*********')
    hold_limit = text['持股限制']
    hold_stock = get_position(c, c.account, c.account_type)
    if hold_stock.shape[0] > 0:
        print(hold_stock, '************')
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
    buy_df = cacal_diurnal_cycle(c)
    print('交易股票池*************')
    buy_df['交易状态'] = '未买'
    print(buy_df)
    if buy_df.shape[0] > 0:
        def select_data(stock):
            if str(stock) in hold_stock_list:
                return '持股超过限制'
            else:
                return '没有持股'

        buy_df['持股检查'] = buy_df['证券代码'].apply(select_data)
        buy_df = buy_df[buy_df['持股检查'] == '没有持股']
    sell_df = get_sell_stock_data(c)
    sell_df['交易状态'] = '未卖'
    if sell_df.shape[0] > 0:
        sell_df['证券代码'] = sell_df['证券代码'].apply(lambda x: '0' * (6 - len(str(x))) + str(x))
        sell_stock_list = sell_df['证券代码'].tolist()
        sell_amount = len(sell_stock_list)
    else:
        sell_amount = 0
    print('卖出股票**********************')
    print(sell_df)
    av_buy = (hold_limit - hold_amount) + sell_amount
    if av_buy >= hold_limit:
        av_buy = hold_limit
    else:
        av_buy = av_buy
    buy_df = buy_df[:av_buy]
    print('买入的标的***************************')
    print(buy_df)
    return buy_df, sell_df


def run_tarder_func(c):
    '''
    运行交易函数
    '''
    trader_models = text['交易模式']
    fix_value = text['固定交易金额']
    fix_amount = text['固定交易金额']
    sep_fix_value = text['特殊交易标的固定交易金额']
    sep_fix_amount = text['特殊交易标的固定交易数量']
    sep_stock_list = text['特殊交易标的']
    if check_is_trader_date_1():
        # 先卖在买入
        buy_df, sell_df = get_buy_sell_stock_data(c)
        if sell_df.shape[0] > 0:
            for stock, hold_amount, av_amount in zip(sell_df['证券代码'], sell_df['持仓量'], sell_df['可用数量']):
                try:
                    if av_amount >= 10:
                        print(
                            '{} 持有数量{} 可以数量{}大于0 卖出数量{}'.format(stock, hold_amount, av_amount, av_amount))
                        passorder(24, 1101, c.account, stock, 5, 0, av_amount, '', 1, '', c)
                    else:
                        print('{} 持有数量{} 可以数量{}等于0 卖出数量{} 不交易'.format(stock, hold_amount, av_amount,
                                                                                       av_amount))
                except:
                    print('{}卖出有问题'.format(stock))
        else:
            print('没有卖出的数据')
        # 买入
        if buy_df.shape[0] > 0:
            for stock in buy_df['证券代码'].tolist():
                if stock in sep_stock_list:
                    print('{}在特殊标的里面*********'.format(stock))
                    fix_value = sep_fix_value
                    volume = sep_fix_amount
                else:
                    fix_value = text['固定交易金额']
                    volume = fix_amount
                print(stock, fix_value)
                if trader_models == '金额':
                    print('{}金额交易模式*******'.format(stock))
                    tader_type, amount, price = order_stock_value(c, c.account, c.account_type, stock, fix_value, 'buy')
                    print(tader_type, amount, price)
                    if tader_type == 'buy' and amount >= 10:
                        passorder(23, 1101, c.account, str(stock), 5, 0, amount, '', 1, '', c)
                        # passorder(23, 1101, c.account, str('513100.SH'), 5, 0, 100, '',1,'',c)
                        print('{} 最新价格 买入{} 元'.format(stock, fix_value))
                    else:
                        print('{}金额交易模式买入不了*******'.format(stock))
                else:
                    print('{}数量交易模式*******'.format(stock))
                    passorder(23, 1101, c.account, str(stock), 5, 0, volume, '', 1, '', c)
                    print('{} 最新价格 买入{} 数量'.format(stock, volume))

        else:
            print('没有买入数据')
    else:
        print('{} 目前不少交易时间'.format(datetime.now()))


def run_get_mi_pulse_trader(c):
    '''
    运行脉冲模块
    '''
    sell_limit = text['向上脉冲']
    buy_limit = text['向下脉冲']
    n = text['脉冲时间']
    fix_value = text['固定交易金额']
    hold_stock = get_position(c, c.account, c.account_type)
    # hold_stock['证券代码']='600031.SH'
    if check_is_trader_date_1():
        if hold_stock.shape[0] > 0:
            hold_stock = hold_stock[hold_stock['持仓量'] >= 10]
            if hold_stock.shape[0] > 0:
                for stock, hold_amount, av_amount in zip(hold_stock['证券代码'], hold_stock['持仓量'],
                                                         hold_stock['可用数量']):
                    try:
                        stats = get_mi_pulse_trader(c, stock=stock, n=n, x1=sell_limit, x2=buy_limit)
                        if stats == 'sell':
                            if av_amount >= 10:
                                print('{} 持有数量{} 可以数量{}大于0 卖出数量{}'.format(stock, hold_amount, av_amount,
                                                                                        av_amount))
                                passorder(24, 1101, c.account, stock, 5, 0, av_amount, '', 1, '', c)
                            else:
                                print('{} 持有数量{} 可以数量{}等于0 卖出数量{} 不交易'.format(stock, hold_amount,
                                                                                               av_amount, av_amount))
                        elif stats == 'buy':
                            print('{} 最新价格 买入{} 元'.format(stock, fix_value))
                            passorder(23, 1102, c.account, stock, 5, -1, fix_value, c)
                        else:
                            print('时间{} 股票{} 不符合脉冲继续分析'.format(datetime.now(), stock))
                    except:
                        print('{} {}分钟脉冲有问题可能不是交易日'.format(stock, datetime.now()))
            else:
                print('{}分钟脉冲没有持股'.format(datetime.now()))
        else:
            print('{}分钟脉冲没有持股'.format(datetime.now()))
    else:
        print('时间{} 符合脉冲不是交易时间'.format(datetime.now()))


def get_mi_pulse_trader(c, stock='603496.SH', n=10, x1=2, x2=-40):
    '''
    分钟脉冲分析
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
    print('股票{} {}分钟 涨跌幅{}'.format(stock, n, zdf))
    if zdf >= x1:
        print('时间{} 卖出{} 触发 {}分钟的脉冲 {} 目前涨跌幅{},脉冲涨跌幅{}'.format(datetime.now(), stock, n, x1, zdf_1,
                                                                                    zdf))
        return 'sell'
    elif zdf <= x2:
        print('时间{} 买入{} 触发 {}分钟的脉冲 {} 目前涨跌幅{} 脉冲涨跌幅{}'.format(datetime.now(), stock, n, x1, zdf_1,
                                                                                    zdf))
        return 'buy'
    else:
        print('时间{} {} 没有触发 {}分钟的脉冲 目前涨跌幅{} 脉冲涨跌幅{} '.format(datetime.now(), stock, n, zdf_1, zdf))
        return ''


def reverse_repurchase_of_treasury_bonds_1(c, buy_ratio=1):
    '''
    国债逆回购1,新的函数
    购买比例buy_ratio
    '''
    # 对交易回调进行订阅，订阅后可以收到交易主推，返回0表示订阅成功
    account = get_account(c, c.account, c.account_type)
    print(account)
    av_cash = account['可用金额']
    av_cash = float(av_cash)
    av_cash = av_cash * buy_ratio
    # stock_code_sh = '204001.SH'
    # 统一用深圳
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
    # 下单的数量要是1000
    amount = int(av_cash / 1000)
    # 想下取整1000的倍数
    amount = math.floor(amount / 10) * 100
    # 借出钱sell
    print('开始逆回购***********')
    if amount > 0:
        sell(c, stock=stock, amount=amount, price=price)
        text = '国债逆回购交易类型 代码{} 价格{} 数量{} 订单编号{}'.format(stock, price, amount, fix_result_order_id)
        return '交易成功', text
    else:
        text = '国债逆回购卖出 标的{} 价格{} 委托数量{}小于0有问题'.format(stock, price, amount)
        print('账户没有可以的钱@@@@@@@@@@@@@@@@@@@')
        return '交易失败', text


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


def cacal_limit(c, stock):
    '''
    计算涨跌幅
    '''
    try:
        tick = c.get_full_tick(stock_code=[stock])
        tick = tick[stock]
        limit = ((tick['lastPrice'] - tick['lastClose']) / tick['lastClose']) * 100
        return limit
    except:
        print('{}涨跌幅计算有问题'.format(stock))
        return -40


def get_price(c, stock):
    '''
    获取最新价格
    '''
    tick = c.get_full_tick(stock_code=[stock])
    tick = tick[stock]
    price = tick['lastPrice']
    return price


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
            data = pd.concat([data, df], ignore_index=True)
    else:
        data = pd.DataFrame()


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
            return 'sell', price, -av_trader_amount
        else:
            print(
                '{} 目标数量{} 持有数量{} 可用数量{}小于 卖出数量{} 卖出全部'.format(stock, target_amount, hold_amount,
                                                                                     av_amount, av_trader_amount))
            return 'sell', price, -av_amount
    else:
        print('{} 目标数量{} 持有数量{}一样不交易'.format(stock, target_amount, hold_amount))
        return '', '', ''


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