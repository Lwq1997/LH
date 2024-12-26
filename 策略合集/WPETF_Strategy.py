#-*- coding: utf-8 -*-
# 如果你的文件包含中文, 请在文件的第一行使用上面的语句指定你的文件编码

# 用到策略及数据相关API请加入下面的语句(如果要兼容研究使用可以使用 try except导入
from kuanke.user_space_api import *
from Strategy import Strategy
from jqdata import *
from kuanke.wizard import *
import numpy as np
import pandas as pd
import talib as tl
from jqlib.technical_analysis import *
import datetime as datet


# 外盘ETF轮动策略
class WPETF_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)
        self.foreign_ETF = [
            '518880.XSHG',  # 黄金
            '513030.XSHG',  # 德国
            '513100.XSHG',  # 纳指
            '164824.XSHE',  # 印度
            '159866.XSHE',  # 日本

            '513500.XSHG',  # 标普500
            '159915.XSHE'  # 创业板100
            # '161716.XSHE',#招商双债
        ]
        self.deltaday = 20  # 上市天数
        self.days = 14  # 计算ATR的序列长度

    def select(self, context):
        log.info(self.name, '--Select函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        self.select_list = self.__get_rank(context)[:self.max_select_count]
        self.print_trade_plan(context, self.select_list)

    def __get_rank(self, context):
        log.info(self.name, '--get_rank函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        etf_pool = self.fun_delNewShare(context, self.foreign_ETF, self.deltaday)
        score_list = []
        if len(etf_pool) == 0:
            # 如果没有合适的 ETF 就买入国债
            etf_pool = self.fun_delNewShare(context, ['511010.XSHG', '511880.XSHG'], self.deltaday)
            if len(etf_pool) == 0:  # 2013年前的测试会出现这种情况
                log.info('ETF_pool 为空！')
            final_list = etf_pool
            return final_list
        for etf in etf_pool:
            try:
                # 计算ATR
                atr = self.getATR(context, etf, period=self.days)
                score_list.append(atr)
            except ValueError as e:
                log.error(e)
                score_list.append(np.nan)
        df = pd.DataFrame(index=etf_pool, data={'ATR': score_list})
        # 删除包含 NaN 值的行
        df = df.dropna()
        df = df.sort_values(by='ATR', ascending=True)
        final_list = list(df.index)
        log.info("——————————————————————————————————")
        for i, etf in enumerate(df.index):
            name = get_security_info(etf).display_name
            log.info("编号:{}. 股票:{}，ATR:{}".format(i + 1, name, df.loc[etf, 'ATR']))
        log.info("——————————————————————————————————")
        return final_list

    # 2 全球ETF 平均真实波幅（ATR）
    def getATR(self, context, stock, period=14):
        log.info(self.name, '--getATR函数--计算', stock, '的 ATR信息--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 获取历史数据
        hData = attribute_history(stock, period + 1, unit='1d',
                                  fields=('close', 'volume', 'open', 'high', 'low'),
                                  skip_paused=True, df=False)
        high = hData['high']
        low = hData['low']
        close = hData['close']
        # 检查并处理 NaN 值
        if any(np.isnan(high)) or any(np.isnan(low)) or any(np.isnan(close)):
            raise ValueError(f"{stock}的历史数据包含NaN(非数字)值。")
        # 计算ATR
        realATR = tl.ATR(high, low, close, timeperiod=period)
        realATR = realATR / close.mean()
        return realATR[-1]

    #############################外盘ETF策略增加通用函数###########################
    # 删除上市少于deltaday天的股票
    def fun_delNewShare(self, context, equity, deltaday):
        log.info(self.name, '--fun_delNewShare函数--',
                 str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        deltaDate = context.current_dt.date() - datet.timedelta(deltaday)
        tmpList = []
        for stock in equity:
            stock_info = get_security_info(stock)
            if stock_info is not None and stock_info.start_date < deltaDate:
                tmpList.append(stock)
        return tmpList
