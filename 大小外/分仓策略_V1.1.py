# 克隆自聚宽文章：https://www.joinquant.com/post/53613
# 标题：小市值多策略交易删减版
# 作者：韭 皇

import math
import datetime
import numpy as np
import pandas as pd
from jqdata import *
import statsmodels.api as sm
from tabulate import tabulate
from sklearn.cluster import KMeans
from collections import OrderedDict
from jqfactor import get_factor_values

global_sold_stock_record = {}  # 全局卖出记录
EMPTY_POSITION_STRATEGIES = [2]  # 1、4月需要空仓的子策略


def initialize(context):
    set_benchmark("000300.XSHG")
    set_option("avoid_future_data", True)
    set_option("use_real_price", True)
    log.set_level("order", "error")
    strategy_configs = [
        (0, "现金", None, 0),
        (1, "破净", pj_Strategy, 0.1),
        (2, "微盘", wp_Strategy, 0.6),
        (3, "全天", qt_Strategy, 0.3),
        (4, "核心", hx_Strategy, 0)
    ]

    g.portfolio_value_proportion = [config[3] for config in strategy_configs]
    set_subportfolios(
        [SubPortfolioConfig(context.portfolio.starting_cash * prop, "stock") for prop in g.portfolio_value_proportion])

    # 初始化策略
    g.strategys = {}
    for index, name, strategy_class, proportion in strategy_configs:
        if strategy_class:
            subportfolio_index = int(index)
            g.strategys[name] = strategy_class(context, subportfolio_index=subportfolio_index, name=name)

    # 设置调仓
    run_monthly(balance_subportfolios, 1, "9:02")  # 资金平衡
    run_daily(record_all_strategies_daily_value, "15:00")  # 记录收益

    # 破净策略调仓设置
    if g.portfolio_value_proportion[1] > 0:
        run_daily(prepare_pj_strategy, "9:03")
        run_monthly(adjust_pj_strategy, 1, "9:30")
        run_daily(check_pj_limit_up, "14:00")
        run_daily(check_pj_limit_up, "14:50")

    # 微盘策略调仓设置
    if g.portfolio_value_proportion[2] > 0:
        run_daily(prepare_wp_strategy, "9:06")
        run_weekly(adjust_wp_strategy, 1, "10:50")
        run_daily(check_wp_limit_up, "14:30")
    # 全天策略调仓设置
    if g.portfolio_value_proportion[3] > 0:
        run_monthly(adjust_qt_strategy, 1, "10:00")

    # 核心策略调仓设置
    if g.portfolio_value_proportion[4] > 0:
        run_daily(adjust_hx_strategy, "10:05")


# 破净策略
def prepare_pj_strategy(context): g.strategys["破净"].prepare(context)


def adjust_pj_strategy(context): g.strategys["破净"].adjust(context)


def check_pj_limit_up(context):
    sold_stocks = g.strategys["破净"].check(context)
    if sold_stocks: g.strategys["破净"].buy_after_sell(context, sold_stocks)


def prepare_wp_strategy(context): g.strategys["微盘"].prepare(context)


def adjust_wp_strategy(context): g.strategys["微盘"].adjust(context)


def check_wp_limit_up(context):
    sold_stocks = g.strategys["微盘"].check(context)
    if sold_stocks: g.strategys["微盘"].buy_after_sell(context, sold_stocks)


# 全天策略
def adjust_qt_strategy(context): g.strategys["全天"].adjust(context)


# 核心策略
def adjust_hx_strategy(context): g.strategys["核心"].adjust(context)


# 中央数据管理器==========================================================
class DataManager:
    def __init__(self, max_cache_size=100):
        self.cache = OrderedDict()
        self.max_cache_size = max_cache_size

    def get_data(self, key, func, *args, **kwargs):
        if key not in self.cache:
            if len(self.cache) >= self.max_cache_size:
                self.cache.popitem(last=False)
            self.cache[key] = func(*args, **kwargs)
        self.cache.move_to_end(key)
        return self.cache[key]


data_manager = DataManager()


# 过滤条件==========================================================
# 过滤20天内卖出的股票
def filter_recently_sold(stocks, current_date):
    return [stock for stock in stocks if
            stock not in global_sold_stock_record or (current_date - global_sold_stock_record[stock]).days >= 20]


# 行业股票数量限制
def filter_stocks_by_industry(stocks, industry_info, max_industry_stocks):
    selected_stocks = []
    industry_count = {}
    for stock in stocks:
        industry = industry_info[stock]
        if industry not in industry_count:
            industry_count[industry] = 0
        if industry_count[industry] < max_industry_stocks:
            selected_stocks.append(stock)
            industry_count[industry] += 1
    return selected_stocks


# 1、4月空仓择时==========================================================
def empty_position_in_jan_apr(context, strategy):
    current_month = context.current_dt.month
    if current_month in [1, 4]:
        index = int(strategy.subportfolio_index)
        subportfolio = context.subportfolios[index]
        for security in list(subportfolio.long_positions.keys()):
            strategy.close_position(security)
        strategy.hold_list = []
        return True
    return False


# 资金平衡函数==========================================================
def balance_subportfolios(context):
    for i in range(1, len(g.portfolio_value_proportion)):
        target = g.portfolio_value_proportion[i] * context.portfolio.total_value
        value = context.subportfolios[i].total_value
        deviation = abs((value - target) / target) if target != 0 else 0
        if deviation > 0.2:
            if context.subportfolios[i].available_cash > 0 and target < value:
                transfer_cash(from_pindex=i, to_pindex=0,
                              cash=min(value - target, context.subportfolios[i].available_cash))
            if target > value and context.subportfolios[0].available_cash > 0:
                transfer_cash(from_pindex=0, to_pindex=i,
                              cash=min(target - value, context.subportfolios[0].available_cash))


# 记录收益==========================================================
def record_all_strategies_daily_value(context):
    for strategy in g.strategys.values():
        strategy.record_daily_value(context)


# 策略类基类---------------------------------------------------------------
class Strategy:
    def __init__(self, context, subportfolio_index, name):
        self.subportfolio_index = subportfolio_index
        self.name = name
        self.stock_sum = 1
        self.hold_list = []
        self.limit_up_list = []
        self.portfolio_value = pd.DataFrame(columns=['date', 'total_value'])
        self.starting_cash = None
        self.sold_stock_record = {}
        self.exclude_days = 20  # 默认值，可在子策略中覆盖
        self.max_industry_stocks = 1  # 默认值，可在子策略中覆盖
        self.min_volume = 5000  # 默认最小交易量

    def filter_basic_stock(self, context, stock_list):
        current_data = get_current_data()
        stock_list = [
            stock for stock in stock_list
            if not (
                    stock.startswith(('3', '68', '4', '8'))
                    or current_data[stock].paused
                    or current_data[stock].is_st
                    or 'ST' in current_data[stock].name
                    or '*' in current_data[stock].name
                    or '退' in current_data[stock].name
                    or current_data[stock].day_open == current_data[stock].high_limit
                    or current_data[stock].day_open == current_data[stock].low_limit
                    or current_data[stock].last_price >= current_data[stock].high_limit * 0.97
                    or current_data[stock].last_price <= current_data[stock].low_limit * 1.04
                    or (context.current_dt.date() - get_security_info(stock).start_date).days < 365
            ) and current_data[stock].last_price < 30
        ]
        current_date = context.current_dt.date()
        stock_list = filter_recently_sold(stock_list, current_date)  # 过滤20天卖出的
        return stock_list

    def getStockIndustry(self, stocks):
        return pd.Series({stock: info["sw_l1"]["industry_name"] for stock, info in data_manager.get_data(
            f"industry_{stocks}", get_industry, stocks
        ).items() if "sw_l1" in info})

    def record_daily_value(self, context):
        subportfolio = context.subportfolios[self.subportfolio_index]
        new_data = {'date': context.current_dt.date(), 'total_value': subportfolio.total_value}
        self.portfolio_value = self.portfolio_value.append(new_data, ignore_index=True)
        if self.starting_cash is None:
            self.starting_cash = subportfolio.total_value
        if self.starting_cash == 0:
            returns = 0
        else:
            returns = (subportfolio.total_value / self.starting_cash - 1) * 100
        rounded_returns = round(returns, 1)
        record(**{self.name + '': rounded_returns})

    def _prepare(self, context):
        self.hold_list = list(context.subportfolios[self.subportfolio_index].long_positions.keys())
        if self.hold_list:
            df = get_price(self.hold_list, end_date=context.previous_date, frequency="daily",
                           fields=["close", "high_limit"],
                           count=1, panel=False, fill_paused=False)
            self.limit_up_list = list(df[df["close"] == df["high_limit"]].code)
        else:
            self.limit_up_list = []

    def _check(self, context):
        self._prepare(context)
        sold_stocks = []
        if self.limit_up_list:
            current_data = get_current_data()
            for stock in self.limit_up_list:
                if current_data[stock].last_price < current_data[stock].high_limit:
                    if self.close_position(stock):
                        sold_stocks.append(stock)
                        self.sold_stock_record[stock] = context.current_dt.date()
                        # 更新全局卖出记录
                        global_sold_stock_record[stock] = context.current_dt.date()
        return sold_stocks

    def _adjust(self, context, target):
        subportfolio = context.subportfolios[self.subportfolio_index]
        for security in self.hold_list:
            if security not in target and security not in self.limit_up_list:
                self.close_position(security)
        position_count = len(subportfolio.long_positions)
        if len(target) == 0 or self.stock_sum - position_count == 0:
            return
        buy_num = min(len(target), self.stock_sum - position_count)
        value = subportfolio.available_cash / buy_num
        for security in target:
            if security not in list(subportfolio.long_positions.keys()):
                if self.open_position(security, value):
                    if position_count == len(target):
                        break

    def order_target_value_(self, security, value):
        return order_target_value(security, value, pindex=self.subportfolio_index)

    def open_position(self, security, value):
        order = self.order_target_value_(security, value)
        return order is not None and order.filled > 0

    def close_position(self, security):
        order = self.order_target_value_(security, 0)
        return order is not None and order.filled == order.amount

    def select(self, context):
        stocks = self.select_stocks(context)
        industry_info = self.getStockIndustry(stocks) if hasattr(self, 'getStockIndustry') else None
        current_date = context.current_dt.date()
        # 使用拆分后的函数
        stocks = filter_recently_sold(stocks, current_date)
        selected_stocks = filter_stocks_by_industry(stocks, industry_info, self.max_industry_stocks)
        if selected_stocks:
            limit = self.stock_sum * 2
            selected_stocks = selected_stocks[:limit]
        return selected_stocks

    def prepare(self, context):
        self._prepare(context)

    def adjust(self, context):
        if self.subportfolio_index in EMPTY_POSITION_STRATEGIES and empty_position_in_jan_apr(context, self):
            return
        target = self.select(context)
        self._adjust(context, target[:self.stock_sum])
        self.record_daily_value(context)

    def check(self, context):
        return self._check(context)

    def buy_after_sell(self, context, sold_stocks):
        if self.subportfolio_index in EMPTY_POSITION_STRATEGIES and context.current_dt.month in [1, 4]:
            return
        target = self.select(context)
        self._adjust(context, target[:self.stock_sum])

    def adjust_portfolio(self, context, target_values):
        subportfolio = context.subportfolios[self.subportfolio_index]
        current_data = get_current_data()
        hold_list = list(subportfolio.long_positions.keys())

        # 清仓被调出的
        for stock in hold_list:
            if stock not in target_values:
                self.close_position(stock)

        # 先卖出
        for stock, target in target_values.items():
            value = subportfolio.long_positions[stock].value if stock in subportfolio.long_positions else 0
            minV = current_data[stock].last_price * 100
            if value - target > self.min_volume and minV < value - target:
                self.order_target_value_(stock, target)

        # 后买入
        for stock, target in target_values.items():
            value = subportfolio.long_positions[stock].value if stock in subportfolio.long_positions else 0
            minV = current_data[stock].last_price * 100
            if (
                    target - value > self.min_volume
                    and minV < subportfolio.available_cash
                    and minV < target - value
            ):
                self.order_target_value_(stock, target)

        self.record_daily_value(context)


# 破净
class pj_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name):
        super().__init__(context, subportfolio_index, name)
        self.stock_sum = 1
        self.exclude_days = 20  # 多久不再买入卖出的股票
        self.max_industry_stocks = 1  # 每个行业最多选的股票数

    def select_stocks(self, context):
        month = context.previous_date.strftime("%Y-%m")
        all_stocks = data_manager.get_data(
            f"all_stocks_{month}",
            get_all_securities,
            "stock",
            date=context.previous_date
        ).index.tolist()
        stocks = self.filter_basic_stock(context, all_stocks)
        q = query(
            valuation.code, valuation.market_cap, valuation.pe_ratio, income.total_operating_revenue
        ).filter(
            valuation.pb_ratio < 1,
            cash_flow.subtotal_operate_cash_inflow > 1e6,
            indicator.adjusted_profit > 1e6,
            indicator.roa > 0.15,
            indicator.inc_net_profit_year_on_year > 0,
            valuation.code.in_(stocks)
        ).order_by(
            indicator.roa.desc()
        )
        stocks = get_fundamentals(q)["code"].tolist()
        return stocks


# 微盘---------------------------------------------------------------
class wp_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name):
        super().__init__(context, subportfolio_index, name)
        self.stock_sum = 6
        self.max_industry_stocks = 1
        self.exclude_days = 20  # 缓冲时间

    def select_stocks(self, context):
        month = context.current_dt.strftime("%Y-%m")
        stocks = data_manager.get_data(
            f"index_stocks_399101_{month}",
            get_index_stocks,
            "399101.XSHE",
            context.current_dt
        )
        stocks = self.filter_basic_stock(context, stocks)
        q = query(
            valuation.code
        ).filter(
            valuation.code.in_(stocks),
        ).order_by(
            valuation.market_cap.asc()  # 根据市值从小到大排序
        )
        stocks = get_fundamentals(q)["code"].tolist()
        return stocks


# 全天---------------------------------------------------------------
class qt_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name):
        super().__init__(context, subportfolio_index, name)
        self.min_volume = 2000
        self.etf_pool = [
            "511010.XSHG",  # 国债ETF
            "518880.XSHG",  # 黄金ETF
            "513100.XSHG",  # 纳指ETF
            "512820.XSHG",  # 银行ETF
            "159980.XSHE",  # 有色金属ETF
            "162411.XSHE",  # 华宝油气LOF
            "159985.XSHE",  # 豆粕ETF
        ]
        self.rates = [0.4, 0.15, 0.15, 0.15, 0.05, 0.05, 0.05]

    def adjust(self, context):
        subportfolio = context.subportfolios[self.subportfolio_index]
        total_value = subportfolio.total_value
        target_values = {etf: total_value * rate for etf, rate in zip(self.etf_pool, self.rates)}
        self.adjust_portfolio(context, target_values)


# 核心
class hx_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name):
        super().__init__(context, subportfolio_index, name)
        self.etf_pool = ['518880.XSHG', '513100.XSHG', '159915.XSHE', '510180.XSHG']
        self.m_days = 25

    def MOM(self, context, etf):
        y = np.log(attribute_history(etf, self.m_days, '1d', ['close'])['close'].values)
        x = np.arange(len(y))
        weights = np.linspace(1, 2, len(y))
        slope = np.polyfit(x, y, 1, w=weights)[0]
        residuals = y - (slope * x + np.polyfit(x, y, 1, w=weights)[1])

        annualized_returns = np.exp(slope * 250) - 1
        r_squared = 1 - (np.sum(weights * residuals ** 2) /
                         np.sum(weights * (y - np.mean(y)) ** 2))
        return annualized_returns * r_squared

    def get_rank(self, context):
        scores = {etf: self.MOM(context, etf) for etf in self.etf_pool}
        df = pd.Series(scores).to_frame('score')
        return df.query('0 < score <= 5').sort_values('score', ascending=False).index.tolist()

    def adjust(self, context):
        sub = context.subportfolios[self.subportfolio_index]
        target_list = self.get_rank(context)[:1]

        # 卖出逻辑
        for etf in set(sub.long_positions) - set(target_list):
            self.close_position(etf)
            print(f'卖出{etf}')

        # 买入逻辑
        need_buy = set(target_list) - set(sub.long_positions)
        if need_buy:
            cash_per = sub.available_cash / len(need_buy)
            for etf in need_buy:
                self.open_position(etf, cash_per) and print(f'买入{etf}')

        self.record_daily_value(context)

