# -*- coding: utf-8 -*-
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


class Rotation_ETF_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name, params):
        super().__init__(context, subportfolio_index, name, params)
        self.etf_pool = [
            "518880.XSHG",  # 黄金ETF（大宗商品）
            "513100.XSHG",  # 纳指100（海外资产）
            "159915.XSHE",  # 创业板100（成长股，科技股，中小盘）
            "510180.XSHG",  # 上证180（价值股，蓝筹股，中大盘）
        ]
        self.m_days = 25  # 动量参考天数
        self.fill_stock = "511880.XSHG"

    def select(self, context):
        log.info(self.name, '--Select函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        score_list = [self.MOM(context, etf) for etf in self.etf_pool]

        df = pd.DataFrame(index=self.etf_pool, data={"score": score_list})
        df = df.sort_values(by="score", ascending=False)
        df = df[(df["score"] > 0) & (df["score"] <= 5)]  # 安全区间，动量过高过低都不好
        target = df.index.tolist()
        if not target:
            target = [self.fill_stock]

        self.select_list = target[: min(len(target), self.max_hold_count)]

        self.print_trade_plan(context, self.select_list)

    def MOM(self, context, etf):
        log.info(self.name, '--MOM函数--', str(context.current_dt.date()) + ' ' + str(context.current_dt.time()))

        # 获取历史数据，包括self.m_days天的每日收盘价
        df = attribute_history(etf, self.m_days, "1d", ["close"])

        # 将收盘价转换为对数形式，以便处理复合效应
        y = np.log(df["close"].values)

        # 时间序列，从0到n-1，表示天数
        n = len(y)
        x = np.arange(n)

        # 线性增加权重，最近的数据权重更高
        weights = np.linspace(1, 2, n)

        # 进行加权线性回归，计算斜率和截距
        slope, intercept = np.polyfit(x, y, 1, w=weights)

        # 计算年化对数收益率
        annualized_log_return = slope * 250

        # 将年化对数收益率转换为年化简单收益率,TODO
        annualized_returns = np.exp(annualized_log_return) - 1

        # 计算回归残差
        residuals = y - (slope * x + intercept)

        # 计算加权R平方
        weighted_ssr = np.sum(weights * residuals ** 2)
        weighted_tss = np.sum(weights * (y - np.mean(y)) ** 2)
        r_squared = 1 - (weighted_ssr / weighted_tss)

        # 返回动量得分，结合年化收益和拟合优度
        return annualized_returns * r_squared
