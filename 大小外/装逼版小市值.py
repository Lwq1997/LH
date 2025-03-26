# 克隆自聚宽文章：https://www.joinquant.com/post/54735
# 标题：小市值装逼版，极致夏普和回撤
# 作者：z i

import math
import datetime
import numpy as np
import pandas as pd
from jqdata import *
from tabulate import tabulate


# 初始化函数
def initialize(context):
    set_benchmark("000001.XSHG")
    set_option("avoid_future_data", True)
    set_option("use_real_price", True)
    log.set_level("order", "error")

    strategy_configs = [
        (0, "现金", None, 0),
        (1, "破净", pj_Strategy, 0),
        (2, "微盘", wp_Strategy, 1),
        (3, "全天", qt_Strategy, 0),
        (4, "核心", hx_Strategy, 0),
    ]

    g.ph_money = [config[3] for config in strategy_configs]
    set_subportfolios([SubPortfolioConfig(context.portfolio.starting_cash * prop, "stock") for prop in
                       g.ph_money])
    g.strategys = {}
    for index, name, strategy_class, proportion in strategy_configs:
        if strategy_class:
            subportfolio_index = int(index)
            g.strategys[name] = strategy_class(context, subportfolio_index=subportfolio_index, name=name)

    # 设置调仓
    # run_daily(all_log, "9:05")  # 日志更新
    run_monthly(all_money, 1, "9:01")  # 资金平衡
    # run_daily(all_record, "15:01")  # 记录收益
    # run_daily(stop_loss, "14:45")  # 微盘止损

    if g.ph_money[1] > 0:  # 破净
        run_daily(prepare_pj_strategy, "9:01")
        run_monthly(adjust_pj_strategy, 1, "9:41")
        run_daily(check_pj_limit_up, "14:41")
    if g.ph_money[2] > 0:  # 微盘
        run_daily(prepare_wp_strategy, "9:02")
        run_weekly(adjust_wp_strategy, 1, "10:50")
        run_daily(check_wp_limit_up, "14:00")
    if g.ph_money[3] > 0:  # 全天
        run_monthly(adjust_qt_strategy, 1, "9:43")
    if g.ph_money[4] > 0:  # 核心
        run_daily(adjust_hx_strategy, "9:44")


# 破净策略
def prepare_pj_strategy(context): g.strategys["破净"].prepare(context)


def adjust_pj_strategy(context):
    target = g.strategys["破净"].select(context)
    g.strategys["破净"].adjust(context, target)


def check_pj_limit_up(context):
    sold_stocks = g.strategys["破净"].check(context)
    if sold_stocks: g.strategys["破净"].buy_after_sell(context, sold_stocks)


# 微盘策略
def prepare_wp_strategy(context): g.strategys["微盘"].prepare(context)


def adjust_wp_strategy(context):
    target = g.strategys["微盘"].select(context)
    g.strategys["微盘"].adjust(context, target)


def check_wp_limit_up(context):
    sold_stocks = g.strategys["微盘"].check(context)
    if sold_stocks: g.strategys["微盘"].buy_after_sell(context, sold_stocks)


# 全天策略
def adjust_qt_strategy(context): g.strategys["全天"].adjust(context)


# 核心策略
def adjust_hx_strategy(context): g.strategys["核心"].adjust(context)


# 策略类基类---------------------------------------------------------------
class Strategy:
    def __init__(self, context, subportfolio_index, name, kongcang_months=None):
        self.subportfolio_index = subportfolio_index
        self.name = name
        self.stock_sum = 1
        self.hold_list = []
        self.limit_up_list = []
        self.portfolio_value = pd.DataFrame(columns=['date', 'total_value'])
        self.starting_cash = None
        self.sold_stock_record = {}  # 卖出记录
        self.min_volume = 5000  # 默认最小交易量
        self.kongcang_months = kongcang_months if kongcang_months is not None else []  # 需要空仓的月份

    def filter_basic_stock(self, context, stock_list):
        """过滤基本股票"""
        current_data = get_current_data()
        stock_list = [
            stock for stock in stock_list
            if not (
                    stock.startswith(('68', '4', '8'))
                    or current_data[stock].paused
                    or current_data[stock].is_st
                    or 'ST' in current_data[stock].name
                    or '*' in current_data[stock].name
                    or '退' in current_data[stock].name
                    or current_data[stock].last_price >= current_data[stock].high_limit * 0.97
                    or current_data[stock].last_price <= current_data[stock].low_limit * 1.04
                    or (context.current_dt.date() - get_security_info(stock).start_date).days < 365
            ) and current_data[stock].last_price < 30
        ]
        current_date = context.current_dt.date()
        stock_list = day20sell(stock_list, current_date, self.sold_stock_record)  # 过滤20天卖出的
        return stock_list

    def prepare(self, context):
        """准备数据"""
        self.hold_list = list(context.subportfolios[self.subportfolio_index].long_positions.keys())
        if self.hold_list:
            df = get_price(self.hold_list, end_date=context.previous_date, frequency="daily",
                           fields=["close", "high_limit"],
                           count=1, panel=False, fill_paused=False)
            self.limit_up_list = list(df[df["close"] == df["high_limit"]].code)
        else:
            self.limit_up_list = []

    def check(self, context):
        ch = list(context.subportfolios[self.subportfolio_index].long_positions.keys())
        ss = []
        # 处理昨日涨停股的断板检测
        ylu = zhangting(context, ch)
        ss.extend(duanban(context, self, ch, ylu))
        # 换手检测
        ss.extend(huanshou(context, self, ch))
        self.hold_list = [s for s in ch if s not in ss]
        return ss

    def adjust(self, context, target):
        """调整仓位"""
        if context.current_dt.month in self.kongcang_months:  # 月份空仓
            self.kongcang(context)
            return
        subportfolio = context.subportfolios[self.subportfolio_index]
        for security in self.hold_list:
            if security not in target and security not in self.limit_up_list:
                self.close_position(context, security)
        position_count = len(subportfolio.long_positions)
        if len(target) == 0 or self.stock_sum - position_count == 0:
            return
        buy_num = min(len(target), self.stock_sum - position_count)
        value = subportfolio.available_cash / buy_num
        for security in target:
            if security not in list(subportfolio.long_positions.keys()):
                if self.open_position(context, security, value):
                    if position_count == len(target):
                        break

    def order_target_value_(self, security, value):
        """调整目标价值"""
        return order_target(security,
                            0 if value == 0 else int(value / get_current_data()[security].last_price / 100) * 100,
                            style=MarketOrderStyle(),
                            pindex=self.subportfolio_index)

    def open_position(self, context, security, value):
        """开仓"""
        subportfolio = context.subportfolios[self.subportfolio_index]
        if subportfolio.available_cash < value:
            return False
        current_data = get_current_data()
        if current_data[security].paused:  # 是否停牌
            return False
        order = self.order_target_value_(security, value)
        return order is not None and order.filled > 0

    def close_position(self, context, security):
        """平仓"""
        subportfolio = context.subportfolios[self.subportfolio_index]
        # 检查持仓是否存在
        if security not in subportfolio.long_positions:
            return False
        position = subportfolio.long_positions[security]
        if position.closeable_amount > 0:
            current_data = get_current_data()
            if current_data[security].paused:  # 是否停牌
                return False
            try:
                order = order_target(security, 0, pindex=self.subportfolio_index)
                if order is not None and order.filled > 0:
                    self.sold_stock_record[security] = context.current_dt.date()
                    return True
            except:
                pass
        return False

    # 选股
    def select(self, context):
        stocks = self.select_stocks(context)
        industry_info = hyxx(stocks)
        current_date = context.current_dt.date()
        stocks = day20sell(stocks, current_date, self.sold_stock_record, self.exclude_days)
        selected_stocks = hyxz(stocks, industry_info, self.max_industry_stocks)
        if selected_stocks:
            limit = self.stock_sum * 3
            selected_stocks = selected_stocks[:limit]
        return selected_stocks

    def buy_after_sell(self, context, sold_stocks):
        """卖出后买入"""
        if context.current_dt.month in self.kongcang_months:
            return
        target = self.select(context)
        self.adjust(context, target[:self.stock_sum])

    def adjust_portfolio(self, context, target_values):
        """调整投资组合"""
        subportfolio = context.subportfolios[self.subportfolio_index]
        current_data = get_current_data()
        hold_list = list(subportfolio.long_positions.keys())
        for stock in hold_list:
            if stock not in target_values:
                self.close_position(context, stock)
        for stock, target in target_values.items():
            value = subportfolio.long_positions[stock].value if stock in subportfolio.long_positions else 0
            minV = current_data[stock].last_price * 100
            if value - target > self.min_volume and minV < value - target:
                self.order_target_value_(stock, target)
        for stock, target in target_values.items():
            value = subportfolio.long_positions[stock].value if stock in subportfolio.long_positions else 0
            minV = current_data[stock].last_price * 100
            if (
                    target - value > self.min_volume
                    and minV < subportfolio.available_cash
                    and minV < target - value
            ):
                self.order_target_value_(stock, target)
        shouyi(context, self)

    def kongcang(self, context):
        """清仓"""
        subportfolio = context.subportfolios[self.subportfolio_index]
        for security in list(subportfolio.long_positions.keys()):
            self.close_position(context, security)
        self.hold_list = []


# =================================子策略================================
class pj_Strategy(Strategy):
    """破净策略"""

    def __init__(self, context, subportfolio_index, name):
        super().__init__(context, subportfolio_index, name)
        self.stock_sum = 1
        self.exclude_days = 20  # 卖出屏蔽天数
        self.max_industry_stocks = 1  # 行业限制数量

    def select_stocks(self, context):
        all_stocks = get_all_securities("stock", date=context.previous_date).index.tolist()
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
            indicator.roa.desc()  # ROA排序
        )
        stocks = get_fundamentals(q)["code"].tolist()
        return stocks


class wp_Strategy(Strategy):
    """微盘策略"""

    def __init__(self, context, subportfolio_index, name):
        super().__init__(context, subportfolio_index, name, kongcang_months=[1, 4])
        self.stock_sum = 6
        self.exclude_days = 20  # 卖出屏蔽天数
        self.max_industry_stocks = 1  # 行业限制数量

    def select_stocks(self, context):
        stocks = get_index_stocks("399101.XSHE", context.current_dt)
        stocks = self.filter_basic_stock(context, stocks)
        q = query(
            valuation.code
        ).filter(
            valuation.code.in_(stocks),
        ).order_by(
            valuation.market_cap.asc()  # 市值排序
        )
        stocks = get_fundamentals(q)["code"].tolist()
        return stocks


# 全天
class qt_Strategy(Strategy):
    def __init__(self, context, subportfolio_index, name):
        super().__init__(context, subportfolio_index, name)
        self.min_volume = 2000
        self.etf_pool = {
            "511010.XSHG": {"name": "国债ETF", "rate": 0.4},
            "518880.XSHG": {"name": "黄金ETF", "rate": 0.2},
            "513100.XSHG": {"name": "纳指ETF", "rate": 0.1},
            "510880.XSHG": {"name": "红利ETF", "rate": 0.1},
            "159980.XSHE": {"name": "有色ETF", "rate": 0.05},
            "162411.XSHE": {"name": "油气LOF", "rate": 0.05},
            "159985.XSHE": {"name": "豆粕ETF", "rate": 0.05},
            "513030.XSHG": {"name": "德国ETF", "rate": 0.05},
        }
        self.rates = [etf["rate"] for etf in self.etf_pool.values()]

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
        return df.query('-1 < score <= 999').sort_values('score', ascending=False).index.tolist()

    def adjust(self, context):
        sub = context.subportfolios[self.subportfolio_index]
        target_list = self.get_rank(context)[:1]
        for etf in set(sub.long_positions) - set(target_list):
            self.close_position(context, etf)
        need_buy = set(target_list) - set(sub.long_positions)
        if need_buy:
            cash_per = sub.available_cash / len(need_buy)
            for etf in need_buy:
                self.open_position(context, etf, cash_per)


# =================================公共函数================================
# 日志
def all_log(context):
    for strategy_name in g.strategys:
        strategy = g.strategys[strategy_name]
        subportfolio = context.subportfolios[strategy.subportfolio_index]
        if subportfolio.long_positions and strategy_name in g.strategys:
            if not isinstance(strategy, (qt_Strategy, hx_Strategy)):
                stocks = strategy.select(context)
                if stocks:
                    log_content = '\n'.join([
                        f"{i + 1:<3} {stock:<8} {get_security_info(stock).display_name:<8} "
                        f"{get_industry(stock, date=None).get(stock, {}).get('sw_l1', {}).get('industry_name', '未知行业'):<15} "
                        f"{'💰' if stock in subportfolio.long_positions else '':<2}"
                        for i, stock in enumerate(stocks)
                    ])
                    log.info(f"\n{'-' * 20}{strategy_name}{'-' * 20}\n{log_content}")


# 资金平衡
def all_money(context):
    for i in range(1, len(g.ph_money)):
        target = g.ph_money[i] * context.portfolio.total_value
        value = context.subportfolios[i].total_value
        deviation = abs((value - target) / target) if target != 0 else 0
        if deviation > 0.2:
            if context.subportfolios[i].available_cash > 0 and target < value:
                transfer_cash(from_pindex=i, to_pindex=0,
                              cash=min(value - target, context.subportfolios[i].available_cash))
            if target > value and context.subportfolios[0].available_cash > 0:
                transfer_cash(from_pindex=0, to_pindex=i,
                              cash=min(target - value, context.subportfolios[0].available_cash))


# 记录收益
def all_record(context):
    for strategy in g.strategys.values():
        shouyi(context, strategy)


# 记录每日收益
def shouyi(context, strategy):
    subportfolio = context.subportfolios[strategy.subportfolio_index]
    total_value = subportfolio.total_value
    if strategy.starting_cash is None:
        strategy.starting_cash = total_value
    returns = 0 if strategy.starting_cash == 0 else (total_value / strategy.starting_cash - 1) * 100
    rounded_returns = round(returns, 1)
    record(**{strategy.name + '': rounded_returns})
    new_row = pd.DataFrame({'date': [context.current_dt.date()], 'total_value': [total_value]})
    strategy.portfolio_value = pd.concat([strategy.portfolio_value, new_row], ignore_index=True)


# 过滤最近卖出的股票
def day20sell(stocks, current_date, sold_stock_record, exclude_days=20):
    return [stock for stock in stocks if
            stock not in sold_stock_record or (current_date - sold_stock_record[stock]).days >= exclude_days]


# 获取股票行业信息
def hyxx(stocks):
    industry_data = get_industry(stocks, date=None)
    return pd.Series(
        {stock: industry_data[stock]['sw_l1']['industry_name'] if 'sw_l1' in industry_data[stock] else None for stock in
         stocks})


# 限制每个行业股票的数量
def hyxz(stocks, industry_info, max_industry_stocks):
    if industry_info is None:
        return stocks
    counts = {}
    result = []
    for stock in stocks:
        industry = industry_info[stock]
        if counts.get(industry, 0) < max_industry_stocks:
            result.append(stock)
            counts[industry] = counts.get(industry, 0) + 1
    return result


# 判断今天是否跳过月份
def kongcang(context):
    month = context.current_dt.month
    if month in g.pass_months:
        return False
    else:
        return True


# 换手率计算
def huanshoulv(context, stock, is_avg=False):
    if is_avg:
        # 计算平均换手率
        start_date = context.current_dt - datetime.timedelta(days=20)
        end_date = context.previous_date
        df_volume = get_price(stock, start_date=start_date, end_date=end_date, frequency='daily', fields=['volume'])
        df_cap = get_valuation(stock, end_date=end_date, fields=['circulating_cap'], count=1)
        circulating_cap = df_cap['circulating_cap'].iloc[0] if not df_cap.empty else 0
        if circulating_cap == 0:
            return 0.0
        df_volume['turnover_ratio'] = df_volume['volume'] / (circulating_cap * 10000)
        return df_volume['turnover_ratio'].mean()
    else:
        # 计算实时换手率
        date_now = context.current_dt
        df_vol = get_price(stock, start_date=date_now.date(), end_date=date_now, frequency='1m', fields=['volume'],
                           skip_paused=False, fq='pre', panel=True, fill_paused=False)
        volume = df_vol['volume'].sum()
        date_pre = context.current_dt - datetime.timedelta(days=1)
        df_circulating_cap = get_valuation(stock, end_date=date_pre, fields=['circulating_cap'], count=1)
        circulating_cap = df_circulating_cap['circulating_cap'][0]
        turnover_ratio = volume / (circulating_cap * 10000)
        return turnover_ratio


# 止损函数
def stop_loss(context):
    # 定义不同策略对应的指数、跌幅阈值以及个股跌幅阈值
    strategy_info = {
        "破净": ("000300.XSHG", 0.1, 0.08),
        "微盘": ("399303.XSHE", 0.05, 0.1)
    }

    for strategy_name, (index_code, index_drop_threshold, stock_drop_threshold) in strategy_info.items():
        strategy = g.strategys.get(strategy_name)
        if not strategy:
            continue

        # 检查策略对应的子组合是否有持仓
        subportfolio = context.subportfolios[strategy.subportfolio_index]
        if not subportfolio.long_positions:
            continue

        # 计算指数日内最高和当前价格
        index_data = get_price(index_code, start_date=context.current_dt.date(), end_date=context.current_dt,
                               frequency='1m', fields=['high', 'close'], skip_paused=False, fq='pre', panel=False)
        if not index_data.empty:
            index_high = index_data['high'].max()
            index_current = index_data['close'].iloc[-1]
            index_drop = (index_high - index_current) / index_high
            if index_drop > index_drop_threshold:
                # 指数下跌超过阈值，清仓对应策略
                strategy.kongcang(context)
                log.info(f"【{strategy_name}】因{index_code}指数下跌超过{index_drop_threshold * 100}%清仓📉")

        # 计算持仓个股日内最高和当前价格
        for stock in subportfolio.long_positions:
            stock_data = get_price(stock, start_date=context.current_dt.date(), end_date=context.current_dt,
                                   frequency='1m', fields=['high', 'close'], skip_paused=False, fq='pre', panel=False)
            if not stock_data.empty:
                stock_high = stock_data['high'].max()
                stock_current = stock_data['close'].iloc[-1]
                stock_drop = (stock_high - stock_current) / stock_high
                if stock_drop > stock_drop_threshold:
                    # 个股下跌超过阈值，清仓个股并重新调仓
                    if strategy.close_position(context, stock):
                        log.info(f"【{strategy_name}】{stock} 因下跌超过{stock_drop_threshold * 100}%清仓🚨")
                        target = strategy.select(context)
                        strategy.adjust(context, target[:strategy.stock_sum])


# 获取昨日涨停股
def zhangting(context, ch):
    if ch:
        df = get_price(ch, end_date=context.previous_date, frequency="daily", fields=["close", "high_limit"], count=1,
                       panel=False, fill_paused=False)
        return df[df["close"] == df["high_limit"]].code.tolist()
    return []


# 处理昨日涨停股的断板检测
def duanban(context, strategy, ch, ylu):
    cd = get_current_data()
    to_sell = [s for s in ch if s in ylu and cd[s].last_price < cd[s].high_limit * 0.997]
    sold = [s for s in to_sell if strategy.close_position(context, s)]
    for s in sold:
        ch.remove(s)
    return sold


# 换手检测
def huanshou(context, strategy, ch):
    ss = []
    cd = get_current_data()
    thresh = {pj_Strategy: (0.001, 0.1), wp_Strategy: (0.003, 0.1)}
    if type(strategy) not in thresh: return ss
    shrink, expand = thresh[type(strategy)]
    for s in ch:
        if cd[s].last_price >= cd[s].high_limit * 0.997: continue
        rt = huanshoulv(context, s, False)
        avg = huanshoulv(context, s, True)
        if avg == 0: continue
        r = rt / avg
        action, icon = '', ''
        if avg < 0.003:
            action, icon = '缩量', '❄️'
        elif rt > expand and r > 2:
            action, icon = '放量', '🔥'
        if action:
            log.info(
                f"【{strategy.name}】{action} {s} {get_security_info(s).display_name} 换手率:{rt:.2%}→均:{avg:.2%} 倍率:{r:.1f}x {icon}")
            if strategy.close_position(context, s):
                ss.append(s)
                strategy.sold_stock_record[s] = context.current_dt.date()
    return ss