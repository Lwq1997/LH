# 克隆自聚宽文章：https://www.joinquant.com/post/51758
# 标题：多策略组合2.0（高手续费，近五年年化30%，回撤7%）
# 作者：O_iX

from jqdata import *
from jqfactor import get_factor_values
import datetime
import math
import numpy as np
import pandas as pd


def initialize(context):
    set_benchmark("515080.XSHG")
    set_option("avoid_future_data", True)
    set_option("use_real_price", True)
    log.info("初始化策略")
    set_slippage(FixedSlippage(0.02), type="stock")
    set_order_cost(
        OrderCost(open_tax=0, close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5),
        type="stock")
    set_order_cost(OrderCost(open_tax=0, close_tax=0, open_commission=0, close_commission=0, min_commission=0),
                   type="mmf")
    g.strategys = {}
    g.portfolio_value_proportion = [0, 0.3, 0.5, 0.2]
    set_subportfolios([
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[0], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[1], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[2], 'stock'),
        SubPortfolioConfig(context.portfolio.starting_cash * g.portfolio_value_proportion[3], 'stock')
    ])
    jsg_strategy = JSG_Strategy(context, 1, "搅屎棍策略")
    all_day_strategy = All_Day_Strategy(context, 2, "全天候策略")
    rotation_etf_strategy = Rotation_ETF_Strategy(context, 3, "核心资产轮动策略")
    g.strategys[jsg_strategy.name] = jsg_strategy
    g.strategys[all_day_strategy.name] = all_day_strategy
    g.strategys[rotation_etf_strategy.name] = rotation_etf_strategy
    run_monthly(balance_subportfolios, 1, "9:00")
    if g.portfolio_value_proportion[1] > 0:
        run_daily(jsg_prepare, "9:05")
        run_weekly(jsg_adjust, 1, "9:31")
        run_daily(jsg_check, "14:50")
    if g.portfolio_value_proportion[2] > 0:
        run_monthly(all_day_adjust, 1, "9:40")
    if g.portfolio_value_proportion[3] > 0:
        run_daily(rotation_etf_adjust, "9:32")


def balance_subportfolios(context):
    length = len(g.portfolio_value_proportion)
    print("每月平衡子账户仓位")
    print(
        "调整前：" + str([context.subportfolios[i].total_value / context.portfolio.total_value for i in range(length)]))
    for i in range(1, length):
        target = g.portfolio_value_proportion[i] * context.portfolio.total_value
        value = context.subportfolios[i].total_value
        if context.subportfolios[i].available_cash > 0 and target < value:
            transfer_cash(from_pindex=i, to_pindex=0, cash=min(value - target, context.subportfolios[i].available_cash))
    for i in range(1, length):
        target = g.portfolio_value_proportion[i] * context.portfolio.total_value
        value = context.subportfolios[i].total_value
        if target > value and context.subportfolios[0].available_cash > 0:
            transfer_cash(from_pindex=0, to_pindex=i, cash=min(target - value, context.subportfolios[0].available_cash))
    print(
        "调整后：" + str([context.subportfolios[i].total_value / context.portfolio.total_value for i in range(length)]))


def jsg_prepare(context):
    g.strategys["搅屎棍策略"].prepare(context)


def jsg_check(context):
    g.strategys["搅屎棍策略"].check(context)


def jsg_adjust(context):
    g.strategys["搅屎棍策略"].adjust(context)


def all_day_adjust(context):
    g.strategys["全天候策略"].adjust(context)


def rotation_etf_adjust(context):
    g.strategys["核心资产轮动策略"].adjust(context)


class Strategy:
    def __init__(self, context, subportfolio_index, name):
        self.subportfolio_index = subportfolio_index
        self.name = name
        self.stock_sum = 1
        self.hold_list = []
        self.limit_up_list = []
        self.fill_stock = "511880.XSHG"

    def _prepare(self, context):
        self.hold_list = list(context.subportfolios[self.subportfolio_index].long_positions.keys())
        df = get_price(self.hold_list, end_date=context.previous_date, frequency="daily",
                       fields=["close", "high_limit"], count=1, panel=False, fill_paused=False)
        df = df[df["close"] == df["high_limit"]]
        self.limit_up_list = list(df.code)

    def _check(self, context):
        if self.limit_up_list:
            current_data = get_current_data()
            for stock in self.limit_up_list:
                if current_data[stock].last_price < current_data[stock].high_limit:
                    log.info("[%s]涨停打开，卖出" % stock)
                    self.close_position(stock)
                else:
                    log.info("[%s]涨停，继续持有" % stock)

    def _adjust(self, context, target):
        subportfolio = context.subportfolios[self.subportfolio_index]
        for security in self.hold_list:
            if (security not in target) and (security not in self.limit_up_list):
                self.close_position(security)
        position_count = len(subportfolio.long_positions)
        if len(target) > position_count:
            buy_num = min(len(target), self.stock_sum - position_count)
            value = subportfolio.available_cash / buy_num
            for security in target:
                if security not in list(subportfolio.long_positions.keys()):
                    if self.open_position(security, value):
                        if position_count == len(target):
                            break

    def order_target_value_(self, security, value):
        if value == 0:
            log.debug("卖出全部%s" % security)
        else:
            log.debug("下单%s目标价值%f" % (security, value))
        return order_target_value(security, value, pindex=self.subportfolio_index)

    def open_position(self, security, value):
        order = self.order_target_value_(security, value)
        return order.filled > 0 if order else False

    def close_position(self, security):
        order = self.order_target_value_(security, 0)
        return order.status == OrderStatus.held and order.filled == order.amount if order else False

    def filter_paused_stock(self, stock_list):
        current_data = get_current_data()
        return [stock for stock in stock_list if not current_data[stock].paused]

    def filter_st_stock(self, stock_list):
        current_data = get_current_data()
        return [stock for stock in stock_list if
                not current_data[stock].is_st and "ST" not in current_data[stock].name and "*" not in current_data[
                    stock].name and "退" not in current_data[stock].name]

    def filter_kcbj_stock(self, stock_list):
        return [stock for stock in stock_list if
                not (stock[0] == "4" or stock[0] == "8" or stock[:2] == "68" or stock[0] == "3")]

    def filter_limitup_limitdown_stock(self, context, stock_list):
        current_data = get_current_data()
        return [stock for stock in stock_list if
                stock in context.subportfolios[self.subportfolio_index].long_positions.keys() or (
                            current_data[stock].last_price < current_data[stock].high_limit and current_data[
                        stock].last_price > current_data[stock].low_limit)]

    def filter_new_stock(self, context, stock_list, days):
        yesterday = context.previous_date
        return [stock for stock in stock_list if
                yesterday - get_security_info(stock).start_date >= datetime.timedelta(days)]


class JSG_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name):
        super().__init__(context, subportfolio_index, name)
        self.stock_sum = 6
        self.pass_months = [1, 4]

    def getStockIndustry(self, stocks):
        industry = get_industry(stocks)
        return pd.Series({stock: info["sw_l1"]["industry_name"] for stock, info in industry.items() if "sw_l1" in info})

    def get_market_breadth(self, context):
        yesterday = context.previous_date
        stocks = get_index_stocks("000985.XSHG")
        h = get_price(stocks, end_date=yesterday, frequency="1d", fields=["close"], count=21, panel=False)
        h["date"] = pd.DatetimeIndex(h.time).date
        df_close = h.pivot(index="code", columns="date", values="close").dropna(axis=0)
        df_ma20 = df_close.rolling(window=20, axis=1).mean().iloc[:, -1:]
        df_bias = df_close.iloc[:, -1:] > df_ma20
        df_bias["industry_name"] = self.getStockIndustry(stocks)
        df_ratio = ((df_bias.groupby("industry_name").sum() * 100.0) / df_bias.groupby("industry_name").count()).round()
        top_values = df_ratio.loc[:, yesterday].nlargest(1)
        I = top_values.index.tolist()
        print([name for name in I], "全市场宽度：", np.array(df_ratio.sum(axis=0).mean()))
        return I

    def filter(self, context):
        stocks = get_index_stocks("399101.XSHE", context.current_dt)
        stocks = self.filter_kcbj_stock(stocks)
        stocks = self.filter_st_stock(stocks)
        stocks = self.filter_new_stock(context, stocks, 375)
        stocks = self.filter_paused_stock(stocks)
        stocks = get_fundamentals(
            query(valuation.code).filter(valuation.code.in_(stocks), income.np_parent_company_owners > 0,
                                         income.net_profit > 0, income.operating_revenue > 1e8))["code"].tolist()
        stocks = self.filter_limitup_limitdown_stock(context, stocks)
        return stocks[:min(len(stocks), self.stock_sum)]

    def is_empty_month(self, context):
        month = context.current_dt.month
        return month in self.pass_months

    def select(self, context):
        I = self.get_market_breadth(context)
        industries = {"银行I", "有色金属I", "煤炭I", "钢铁I", "采掘I"}
        if not industries.intersection(I) and not self.is_empty_month(context):
            print("开仓")
            L = self.filter(context)
        else:
            print("跑")
            L = [self.fill_stock]
        return L

    def prepare(self, context):
        self._prepare(context)

    def adjust(self, context):
        target = self.select(context)
        self._adjust(context, target)

    def check(self, context):
        self._check(context)


class All_Day_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name):
        super().__init__(context, subportfolio_index, name)
        self.etf_pool = ["511010.XSHG", "518880.XSHG", "513100.XSHG", "515080.XSHG", "159980.XSHE", "162411.XSHE",
                         "159985.XSHE"]
        self.rates = [0.4, 0.2, 0.15, 0.1, 0.05, 0.05, 0.05]

    def adjust(self, context):
        subportfolio = context.subportfolios[self.subportfolio_index]
        targets = {etf: subportfolio.total_value * rate for etf, rate in zip(self.etf_pool, self.rates)}
        if not subportfolio.long_positions:
            for etf, target in targets.items():
                self.order_target_value_(etf, target)
        else:
            for etf, target in targets.items():
                value = subportfolio.long_positions.get(etf, {}).get('value', 0)
                if value != target:
                    self.order_target_value_(etf, target)


class Rotation_ETF_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name):
        super().__init__(context, subportfolio_index, name)
        self.etf_pool = ["518880.XSHG", "513100.XSHG", "159915.XSHE", "510180.XSHG"]
        self.m_days = 25

    def MOM(self, etf):
        df = attribute_history(etf, self.m_days, "1d", ["close"])
        y = np.log(df["close"].values)
        x = np.arange(len(y))
        weights = np.linspace(1, 2, len(y))
        slope, intercept = np.polyfit(x, y, 1, w=weights)
        annualized_returns = math.pow(math.exp(slope), 250) - 1
        residuals = y - (slope * x + intercept)
        r_squared = 1 - (np.sum(weights * residuals ** 2) / np.sum(weights * (y - np.mean(y)) ** 2))
        return annualized_returns * r_squared

    def select(self):
        score_list = [self.MOM(etf) for etf in self.etf_pool]
        df = pd.DataFrame(index=self.etf_pool, data={"score": score_list})
        df = df.sort_values(by="score", ascending=False)
        df = df[(df["score"] > 0) & (df["score"] <= 5)]
        target = df.index.tolist()
        if not target:
            target = [self.fill_stock]
        return target[:min(len(target), self.stock_sum)]

    def adjust(self, context):
        target = self.select()
        self._prepare(context)
        self._adjust(context, target)
